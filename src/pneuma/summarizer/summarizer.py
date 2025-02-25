"""
summarizer.py

This module provides summarization functionality for indexed tables.
"""

import gc
import json
import logging
import math
from collections import defaultdict

import duckdb
import fire
import pandas as pd
import tiktoken
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import TextGenerationPipeline

from pneuma.utils.logging_config import configure_logging
from pneuma.utils.prompting_interface import (
    prompt_openai_llm,
    prompt_pipeline,
    prompt_pipeline_robust,
)
from pneuma.utils.response import Response, ResponseStatus
from pneuma.utils.summary_types import SummaryType
from pneuma.utils.table_status import TableStatus

configure_logging()
logger = logging.getLogger("Summarizer")


class Summarizer:
    def __init__(
        self,
        llm: OpenAI | TextGenerationPipeline,
        embed_model: OpenAI | SentenceTransformer,
        db_path: str,
        max_llm_batch_size: int = 50,
    ):
        """
        Summarizes indexed tables in the database.

        This class provides a method to summarize indexed tables in the database
        to represent them for retrieval purposes.

        ## Attributes
        - **pipe** (`OpenAI | TextGenerationPipeline`): The LLM pipeline for inference.
        - **embedding_model** (`OpenAI | SentenceTransformer`): The model used for
        text embeddings.
        - **db_path** (`str`): Path to the database file for retrieving content
        summaries & context.
        - **MAX_LLM_BATCH_SIZE** (`int`): The upper bound of batch size value to
        explore dynamically for LLM inference.
        - **EMBEDDING_MAX_TOKENS** (`int`): The maximum number of tokens the embedding
        model supports (hard-coded to 512 for local models and 8191 for OpenAI models).
        """
        self.db_path = db_path
        self.pipe = llm
        self.embedding_model = embed_model
        self.MAX_LLM_BATCH_SIZE = max_llm_batch_size

        if isinstance(self.embedding_model, OpenAI):
            self.EMBEDDING_MAX_TOKENS = 8191
        else:
            self.EMBEDDING_MAX_TOKENS = 512

    def summarize(self, table_id: str = None) -> str:
        """
        Summarizes the contents of all unsummarized tables or a specific table
        if `table_id` is provided.

        ## Args
        - **table_id** (`str`): The specific table ID to be summarized.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                if table_id is None or table_id == "":
                    logger.info("Generating summaries for all unsummarized tables")
                    table_ids = [
                        entry[0].replace("'", "''")
                        for entry in connection.sql(
                            f"""SELECT id FROM table_status
                            WHERE status = '{TableStatus.REGISTERED}'"""
                        ).fetchall()
                    ]
                    logger.info(f"Found {len(table_ids)} unsummarized tables")
                else:
                    table_ids = [table_id.replace("'", "''")]

                if len(table_ids) == 0:
                    return Response(
                        status=ResponseStatus.SUCCESS,
                        message="No unsummarized tables found.\n",
                        data={"table_ids": []},
                    ).to_json()
                if len(table_ids) == 1:
                    all_summary_ids = self.__summarize_table_by_id(table_ids[0])
                else:
                    all_summary_ids = self.__batch_summarize_tables(table_ids)

                if len(all_summary_ids) == 0:
                    return Response(
                        status=ResponseStatus.ERROR,
                        message=f"Summarization failed",
                    ).to_json()

                return Response(
                    status=ResponseStatus.SUCCESS,
                    message=f"Total of {len(all_summary_ids)} summaries has been added "
                    f"with IDs: {', '.join([str(summary_id) for summary_id in all_summary_ids])}.\n",
                    data={"table_ids": table_ids, "summary_ids": all_summary_ids},
                ).to_json()
        except Exception as e:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Error connecting to database: {e}",
            ).to_json()

    def __summarize_table_by_id(self, table_id: str) -> list[str]:
        """
        Summarizes the contents of a single table: `table_id`.

        ## Args
        - **table_id** (`str`): The specific table ID to be summarized.

        ## Returns
        - `list[str]`: The database IDs of the resulting summaries for table
        `table_id`.
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                status = connection.sql(
                    f"SELECT status FROM table_status WHERE id = '{table_id}'"
                ).fetchone()[0]
                if status == str(TableStatus.SUMMARIZED) or status == str(
                    TableStatus.DELETED
                ):
                    logger.warning(
                        "Table with ID %s has already been summarized.", table_id
                    )
                    return []

                table_df = connection.sql(f"SELECT * FROM '{table_id}'").to_df()

                narration_summaries = self.__generate_column_narrations(table_df)
                row_samples = self.__generate_row_samples(table_df)

                summary_ids = []

                for narration_summary in narration_summaries:
                    narration_payload = json.dumps({"payload": narration_summary})
                    narration_payload = narration_payload.replace("'", "''")
                    summary_ids.append(
                        connection.sql(
                            f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                            VALUES ('{table_id}', '{narration_payload}', '{SummaryType.COLUMN_NARRATION}')
                            RETURNING id"""
                        ).fetchone()[0]
                    )

                for row_sample in row_samples:
                    row_payload = json.dumps({"payload": row_sample})
                    row_payload = row_payload.replace("'", "''")
                    summary_ids.append(
                        connection.sql(
                            f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                            VALUES ('{table_id}', '{row_payload}', '{SummaryType.ROW_SAMPLE}')
                            RETURNING id"""
                        ).fetchone()[0]
                    )

                connection.sql(
                    f"""UPDATE table_status
                    SET status = '{TableStatus.SUMMARIZED}'
                    WHERE id = '{table_id}'"""
                )

                return summary_ids
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return []

    def __batch_summarize_tables(self, table_ids: list[str]) -> list[str]:
        """
        Summarizes the contents of tables `table_ids`.

        ## Args
        - **table_ids** (`list[str]`): The specific table IDs to be summarized.

        ## Returns
        - `list[str]`: The database IDs of the resulting summaries for the tables.
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                for table_id in table_ids:
                    status = connection.sql(
                        f"SELECT status FROM table_status WHERE id = '{table_id}'"
                    ).fetchone()[0]
                    if status == str(TableStatus.SUMMARIZED) or status == str(
                        TableStatus.DELETED
                    ):
                        logger.warning(
                            "Table with ID %s has already been summarized.", table_id
                        )
                        table_ids.remove(table_id)

                all_narration_summaries = self.__batch_generate_column_narrations(
                    table_ids
                )
                summary_ids = []

                for table_id, narration_summaries in all_narration_summaries.items():
                    table_df = connection.sql(f"SELECT * FROM '{table_id}'").to_df()
                    row_samples = self.__generate_row_samples(table_df)

                    for narration_summary in narration_summaries:
                        narration_payload = json.dumps({"payload": narration_summary})
                        narration_payload = narration_payload.replace("'", "''")
                        summary_ids.append(
                            connection.sql(
                                f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                                VALUES ('{table_id}', '{narration_payload}', '{SummaryType.COLUMN_NARRATION}')
                                RETURNING id"""
                            ).fetchone()[0]
                        )

                    for row_sample in row_samples:
                        row_payload = json.dumps({"payload": row_sample})
                        row_payload = row_payload.replace("'", "''")
                        summary_ids.append(
                            connection.sql(
                                f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                                VALUES ('{table_id}', '{row_payload}', '{SummaryType.ROW_SAMPLE}')
                                RETURNING id"""
                            ).fetchone()[0]
                        )

                    connection.sql(
                        f"""UPDATE table_status
                        SET status = '{TableStatus.SUMMARIZED}'
                        WHERE id = '{table_id}'"""
                    )

                return summary_ids
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return []

    def __generate_column_narrations(self, df: pd.DataFrame) -> list[str]:
        """Generate column narrations for a single dataframe (for quick local
        testing). This method may be removed in the future."""
        cols = df.columns
        conversations = []
        for col in cols:
            prompt = self.__get_col_narration_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])

        if len(conversations) > 0:
            if isinstance(self.pipe, OpenAI):
                outputs = prompt_openai_llm(
                    llm=self.pipe,
                    conversations=conversations,
                    max_new_tokens=400,
                )
            else:
                outputs = prompt_pipeline(
                    self.pipe,
                    conversations,
                    batch_size=2,
                    context_length=32768,
                    max_new_tokens=400,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            col_narrations: list[str] = []
            for output_idx, output in enumerate(outputs):
                col_narrations.append(
                    f"{cols[output_idx]}: {output[-1]['content']}".strip()
                )

        merged_column_descriptions = self.__block_column_narrations(col_narrations)
        return merged_column_descriptions

    def __batch_generate_column_narrations(
        self, table_ids: list[str]
    ) -> dict[str, list[str]]:
        """
        Generates column narrations for the tables `table_ids`.

        ## Args
        - **table_ids** (`list[str]`): The specific table IDs to be narrated.

        ## Returns
        - `dict[str, list[str]]`: The column narrations of the tables.
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                summaries: dict[str, list[str]] = {}
                conversations = []
                conv_tables = []
                conv_cols = []

                for table_id in table_ids:
                    table_df = connection.sql(f"SELECT * FROM '{table_id}'").to_df()
                    cols = table_df.columns
                    for col in cols:
                        prompt = self.__get_col_narration_prompt(" | ".join(cols), col)
                        conversations.append([{"role": "user", "content": prompt}])
                        conv_tables.append(table_id)
                        conv_cols.append(col)

                if isinstance(self.pipe, OpenAI):
                    optimal_batch_size = 1
                    sorted_indices = list(range(len(conversations)))
                else:
                    optimal_batch_size = self.__get_optimal_batch_size(conversations)
                    sorted_indices = self.__get_special_indices(
                        conversations, optimal_batch_size
                    )

                conversations = [conversations[i] for i in sorted_indices]
                conv_tables = [conv_tables[i] for i in sorted_indices]
                conv_cols = [conv_cols[i] for i in sorted_indices]

                if len(conversations) > 0:
                    if isinstance(self.pipe, OpenAI):
                        outputs = prompt_openai_llm(
                            llm=self.pipe,
                            conversations=conversations,
                            max_new_tokens=400,
                        )
                    else:
                        outputs: list[list[dict[str, str]]] = []
                        max_batch_size = optimal_batch_size
                        same_batch_size_counter = 0
                        for i in tqdm(range(0, len(conversations), optimal_batch_size)):
                            llm_output = prompt_pipeline_robust(
                                self.pipe,
                                conversations[i : i + optimal_batch_size],
                                batch_size=optimal_batch_size,
                                context_length=32768,
                                max_new_tokens=400,
                                temperature=None,
                                top_p=None,
                                top_k=None,
                            )
                            outputs += llm_output[0]

                            if llm_output[1] == optimal_batch_size:
                                same_batch_size_counter += 1
                                if same_batch_size_counter % 10 == 0:
                                    optimal_batch_size = min(
                                        optimal_batch_size + 2, max_batch_size
                                    )
                            else:
                                optimal_batch_size = llm_output[1]
                                same_batch_size_counter = 0

                    col_narrations: dict[str, list[str]] = defaultdict(list)
                    for output_idx, output in enumerate(outputs):
                        col_narrations[conv_tables[output_idx]] += [
                            f"{conv_cols[output_idx]}: {output[-1]['content']}".strip()
                        ]

                for key, value in col_narrations.items():
                    summaries[key] = self.__block_column_narrations(value)

                return summaries
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return {}

    def __get_col_narration_prompt(self, columns: str, column: str) -> str:
        """
        Returns the prompt to narrate a column of a table given other columns
        in the table.

        ## Args
        - **columns** (`str`): A concatenation of `columns`.
        - **column** (`str`): A specific column (part of `columns`) to be narrated.

        ## Returns
        - `str`: The prompt to narrate `column`.
        """
        return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""

    def __get_optimal_batch_size(self, conversations: list[dict[str, str]]) -> int:
        """
        Explores the optimal batch size value (bounded between 1 and
        `MAX_LLM_BATCH_SIZE`) for `conversations` to be set for the LLM pipeline
        using binary search.

        ## Args
        - **conversations** (`list[dict[str, str]]`): The list of prompts to
        narrate columns of tables.

        ## Returns
        - `int`: The optimal batch size.
        """
        max_batch_size = self.MAX_LLM_BATCH_SIZE
        min_batch_size = 1
        logger.info(
            f"Exploring optimal batch size within [{min_batch_size},{max_batch_size}] (binary search)"
        )

        while min_batch_size < max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size) // 2
            logger.info(f"=> Current mid batch size: {mid_batch_size}")
            if self.__is_fit_in_memory(conversations, mid_batch_size):
                min_batch_size = mid_batch_size + 1
            else:
                max_batch_size = mid_batch_size - 1
        optimal_batch_size = min_batch_size
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size

    def __is_fit_in_memory(
        self, conversations: list[dict[str, str]], batch_size: int
    ) -> bool:
        """
        Checks if `conversations` with the given `batch_size` fits in memory when
        running inference using the LLM pipeline.

        ## Args
        - **conversations** (`list[dict[str, str]]`): The list of prompts to
        narrate columns of tables.
        - **batch_size** (`int`): The specific batch size value to test.

        ## Returns
        - `bool`: The `conversations` fit or not in memory.
        """
        special_indices = self.__get_special_indices(conversations, batch_size)
        adjusted_conversations = [conversations[i] for i in special_indices]

        conv_low_idx = len(adjusted_conversations) // 2 - batch_size // 2
        conv_high_dx = conv_low_idx + batch_size
        output = prompt_pipeline(
            self.pipe,
            adjusted_conversations[conv_low_idx:conv_high_dx],
            batch_size=batch_size,
            context_length=32768,
            max_new_tokens=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        torch.cuda.empty_cache()
        gc.collect()

        if output[0][0]["content"] == "":
            del output
            return False
        else:
            del output
            return True

    def __get_special_indices(self, prompts: list[str], batch_size: int) -> list[int]:
        """
        Sorts `prompts` in a specific manner to try to balance the memory load
        for each batch of LLM inferences.

        ## Args
        - **prompts** (`list[str]`): The list of prompts to narrate columns of
        tables.
        - **batch_size** (`int`): The optimal batch size value to be used.

        ## Returns
        - `list[int]`: The "special indices" for the `prompts` given the `batch
        size`.
        """
        # Step 1: Sort the conversations (indices) in decreasing order
        sorted_indices = sorted(
            range(len(prompts)), key=lambda x: len(prompts[x]), reverse=True
        )

        # Step 2: Interleave the indices (longest, shortest, second longest, second shortest, ...)
        final_indices: list[int] = []
        i, j = 0, len(sorted_indices) - 1

        while i <= j:
            if i == j:
                final_indices.append(sorted_indices[i])
                break

            final_indices.append(sorted_indices[i])
            i += 1

            for _ in range(batch_size - 1):
                if i <= j:
                    final_indices.append(sorted_indices[j])
                    j -= 1
                else:
                    break
        return final_indices

    def __block_column_narrations(self, column_narrations: list[str]) -> list[str]:
        """
        Convert column narrations into blocks to try to group multiple narrations
        as much as possible, reducing the amount of embeddings that need to be
        produced.

        ## Args
        - **column_narrations** (`list[str]`): The list of column narrations for
        a set of tables.

        ## Returns
        - `list[str]`: The blocked version of `column_narrations`.
        """
        if isinstance(self.embedding_model, OpenAI):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = self.embedding_model.tokenizer
        merged_column_descriptions = []

        col_idx = 0
        while col_idx < len(column_narrations):
            current_description = column_narrations[col_idx]
            while col_idx + 1 < len(column_narrations):
                combined_description = (
                    current_description + " || " + column_narrations[col_idx + 1]
                )
                if (
                    len(tokenizer.encode(combined_description))
                    < self.EMBEDDING_MAX_TOKENS
                ):
                    current_description = combined_description
                    col_idx += 1
                else:
                    break

            col_idx += 1
            merged_column_descriptions.append(current_description)

        return merged_column_descriptions

    def __generate_row_samples(self, df: pd.DataFrame) -> list[str]:
        """
        Generates row samples for the table `df`. The process is deterministic
        because we set the sampling seed to be the value 0.

        ## Args
        - **df** (`pd.DataFrame`): The specific table to sample rows from.

        ## Returns
        - `list[str]`: The sampled rows
        """
        sample_size = math.ceil(min(len(df), 5))
        selected_df = df.sample(n=sample_size, random_state=0).reset_index(drop=True)

        row_samples = []
        for row_idx, row in selected_df.iterrows():
            formatted_row = " | ".join([f"{col}: {val}" for col, val in row.items()])
            row_samples.append(formatted_row.strip())

        merged_row_samples = self.__block_row_samples(row_samples)
        return merged_row_samples

    def __block_row_samples(self, row_samples: list[str]) -> list[str]:
        """
        Convert row samples into blocks to try to group multiple samples
        as much as possible, reducing the amount of embeddings that need to be
        produced.

        ## Args
        - **row_samples** (`list[str]`): The list of row samples for a set of
        tables.

        ## Returns
        - `list[str]`: The blocked version of `row_samples`.
        """
        if isinstance(self.embedding_model, OpenAI):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = self.embedding_model.tokenizer
        merged_row_samples = []

        row_idx = 0
        while row_idx < len(row_samples):
            current_summary = row_samples[row_idx]
            while row_idx + 1 < len(row_samples):
                combined_summary = current_summary + " || " + row_samples[row_idx + 1]
                if len(tokenizer.encode(combined_summary)) < self.EMBEDDING_MAX_TOKENS:
                    current_summary = combined_summary
                    row_idx += 1
                else:
                    break

            row_idx += 1
            merged_row_samples.append(current_summary)

        return merged_row_samples


if __name__ == "__main__":
    fire.Fire(Summarizer)
