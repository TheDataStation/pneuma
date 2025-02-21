import gc
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import fire
import pandas as pd
import tiktoken
import torch
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_config import configure_logging
from utils.prompting_interface import (prompt_openai_llm, prompt_pipeline,
                                       prompt_pipeline_robust)
from utils.response import Response, ResponseStatus
from utils.storage_config import get_storage_path
from utils.summary_types import SummaryType
from utils.table_status import TableStatus

configure_logging()
logger = logging.getLogger("Summarizer")


class Summarizer:
    def __init__(
        self,
        llm,
        embed_model,        
        db_path: str = os.path.join(get_storage_path(), "storage.db"),
        max_llm_batch_size: int = 50,
    ):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        self.pipe = llm
        self.embedding_model = embed_model
        self.MAX_LLM_BATCH_SIZE = max_llm_batch_size

        if isinstance(self.embedding_model, OpenAI):
            self.EMBEDDING_MAX_TOKENS = 8191
        else:
            self.EMBEDDING_MAX_TOKENS = 512

    def summarize(self, table_id: str = None) -> str:
        if table_id is None or table_id == "":
            logger.info("Generating summaries for all unsummarized tables...")
            table_ids = [
                entry[0].replace("'", "''")
                for entry in self.connection.sql(
                    f"""SELECT id FROM table_status
                    WHERE status = '{TableStatus.REGISTERED}'"""
                ).fetchall()
            ]
            logger.info("Found %d unsummarized tables.", len(table_ids))
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

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Total of {len(all_summary_ids)} summaries has been added "
            f"with IDs: {', '.join([str(summary_id) for summary_id in all_summary_ids])}.\n",
            data={"table_ids": table_ids, "summary_ids": all_summary_ids},
        ).to_json()

    def purge_tables(self) -> str:
        summarized_table_ids = [
            entry[0]
            for entry in self.connection.sql(
                f"SELECT id FROM table_status WHERE status = '{TableStatus.SUMMARIZED}'"
            ).fetchall()
        ]

        for table_id in summarized_table_ids:
            logger.info("Dropping table with ID: %s", table_id)
            # Escape single quotes to avoid breaking the SQL query
            table_id = table_id.replace("'", "''")
            self.connection.sql(f'DROP TABLE "{table_id}"')
            self.connection.sql(
                f"""UPDATE table_status
                SET status = '{TableStatus.DELETED}'
                WHERE id = '{table_id}'"""
            )

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Total of {len(summarized_table_ids)} tables have been purged.\n",
        ).to_json()

    def __summarize_table_by_id(self, table_id: str) -> list[str]:
        # TODO: Handle case if table_id is invalid so status is a NoneType.
        status = self.connection.sql(
            f"SELECT status FROM table_status WHERE id = '{table_id}'"
        ).fetchone()[0]
        if status == str(TableStatus.SUMMARIZED) or status == str(TableStatus.DELETED):
            logger.warning("Table with ID %s has already been summarized.", table_id)
            return []

        table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()

        narration_summaries = self.__generate_column_description(table_df)
        row_summaries = self.__generate_row_summaries(table_df)

        summary_ids = []

        for narration_summary in narration_summaries:
            narration_payload = json.dumps({"payload": narration_summary})
            narration_payload = narration_payload.replace("'", "''")
            summary_ids.append(
                self.connection.sql(
                    f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                    VALUES ('{table_id}', '{narration_payload}', '{SummaryType.NARRATION}')
                    RETURNING id"""
                ).fetchone()[0]
            )

        for row_summary in row_summaries:
            row_payload = json.dumps({"payload": row_summary})
            row_payload = row_payload.replace("'", "''")
            summary_ids.append(
                self.connection.sql(
                    f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                    VALUES ('{table_id}', '{row_payload}', '{SummaryType.ROW_SUMMARY}')
                    RETURNING id"""
                ).fetchone()[0]
            )

        self.connection.sql(
            f"""UPDATE table_status
            SET status = '{TableStatus.SUMMARIZED}'
            WHERE id = '{table_id}'"""
        )

        return summary_ids

    def __batch_summarize_tables(self, table_ids: list[str]) -> list[str]:
        for table_id in table_ids:
            status = self.connection.sql(
                f"SELECT status FROM table_status WHERE id = '{table_id}'"
            ).fetchone()[0]
            if status == str(TableStatus.SUMMARIZED) or status == str(
                TableStatus.DELETED
            ):
                logger.warning(
                    "Table with ID %s has already been summarized.", table_id
                )
                table_ids.remove(table_id)

        all_narration_summaries = self.__batch_generate_column_description(table_ids)
        summary_ids = []

        for table_id, narration_summaries in all_narration_summaries.items():
            table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()
            row_summaries = self.__generate_row_summaries(table_df)

            for narration_summary in narration_summaries:
                narration_payload = json.dumps({"payload": narration_summary})
                narration_payload = narration_payload.replace("'", "''")
                summary_ids.append(
                    self.connection.sql(
                        f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                        VALUES ('{table_id}', '{narration_payload}', '{SummaryType.NARRATION}')
                        RETURNING id"""
                    ).fetchone()[0]
                )

            for row_summary in row_summaries:
                row_payload = json.dumps({"payload": row_summary})
                row_payload = row_payload.replace("'", "''")
                summary_ids.append(
                    self.connection.sql(
                        f"""INSERT INTO table_summaries (table_id, summary, summary_type)
                        VALUES ('{table_id}', '{row_payload}', '{SummaryType.ROW_SUMMARY}')
                        RETURNING id"""
                    ).fetchone()[0]
                )

            self.connection.sql(
                f"""UPDATE table_status
                SET status = '{TableStatus.SUMMARIZED}'
                WHERE id = '{table_id}'"""
            )

        return summary_ids

    def __generate_column_description(self, df: pd.DataFrame) -> list[str]:
        # Used for quick local testing
        # return " description | ".join(df.columns).strip() + " description"

        cols = df.columns
        conversations = []
        for col in cols:
            prompt = self.__get_col_description_prompt(" | ".join(cols), col)
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

        merged_column_descriptions = self.__merge_column_descriptions(col_narrations)
        return merged_column_descriptions

    def __batch_generate_column_description(
        self, table_ids: list[str]
    ) -> dict[str, list[str]]:
        summaries: dict[str, list[str]] = {}
        conversations = []
        conv_tables = []
        conv_cols = []

        for table_id in table_ids:
            table_df = self.connection.sql(f"SELECT * FROM '{table_id}'").to_df()
            cols = table_df.columns
            for col in cols:
                prompt = self.__get_col_description_prompt(" | ".join(cols), col)
                conversations.append([{"role": "user", "content": prompt}])
                conv_tables.append(table_id)
                conv_cols.append(col)

        if isinstance(self.pipe, OpenAI):
            optimal_batch_size = 1
            sorted_indices = list(range(len(conversations)))
        else:
            optimal_batch_size = self.__get_optimal_batch_size(conversations)
            sorted_indices = self.__get_special_indices(conversations, optimal_batch_size)

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
                print(f"Optimal batch size: {optimal_batch_size}")
                for i in tqdm(range(0, len(conversations), optimal_batch_size)):
                    llm_output = prompt_pipeline_robust(
                        self.pipe,
                        conversations[i : i + optimal_batch_size],
                        batch_size=optimal_batch_size,
                        context_length=32768,
                        max_new_tokens=400,
                        temperature=None,
                        top_p=None,
                    )
                    outputs += llm_output[0]

                    if llm_output[1] == optimal_batch_size:
                        same_batch_size_counter += 1
                        if same_batch_size_counter % 10 == 0:
                            optimal_batch_size = min(optimal_batch_size + 2, max_batch_size)
                    else:
                        optimal_batch_size = llm_output[1]
                        same_batch_size_counter = 0

            col_narrations: dict[str, list[str]] = defaultdict(list)
            for output_idx, output in enumerate(outputs):
                col_narrations[conv_tables[output_idx]] += [
                    f"{conv_cols[output_idx]}: {output[-1]['content']}".strip()
                ]

        for key, value in col_narrations.items():
            summaries[key] = self.__merge_column_descriptions(value)

        return summaries

    def __get_col_description_prompt(self, columns: str, column: str):
        return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""

    def __get_optimal_batch_size(self, conversations: list[dict[str, str]]):
        max_batch_size = self.MAX_LLM_BATCH_SIZE
        min_batch_size = 1
        while min_batch_size < max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size) // 2
            if self.__is_fit_in_memory(conversations, mid_batch_size):
                min_batch_size = mid_batch_size + 1
            else:
                max_batch_size = mid_batch_size - 1
        optimal_batch_size = min_batch_size
        return optimal_batch_size

    def __is_fit_in_memory(self, conversations: list[dict[str, str]], batch_size: int):
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

    def __get_special_indices(self, texts: list[str], batch_size: int):
        # Step 1: Sort the conversations (indices) in decreasing order
        sorted_indices = sorted(
            range(len(texts)), key=lambda x: len(texts[x]), reverse=True
        )

        # Step 2: Interleave the indices (longest, shortest, second longest, second shortest, ...)
        final_indices = []
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

    def __merge_column_descriptions(self, column_narrations: list[str]) -> list[str]:
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

    def __generate_row_summaries(self, df: pd.DataFrame) -> list[str]:
        sample_size = math.ceil(min(len(df), 5))
        selected_df = df.sample(n=sample_size, random_state=0).reset_index(drop=True)

        row_summaries = []
        for row_idx, row in selected_df.iterrows():
            formatted_row = " | ".join([f"{col}: {val}" for col, val in row.items()])
            row_summaries.append(formatted_row.strip())

        merged_row_summaries = self.__merge_row_summaries(row_summaries)
        return merged_row_summaries

    def __merge_row_summaries(self, row_summaries: list[str]) -> list[str]:
        if isinstance(self.embedding_model, OpenAI):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = self.embedding_model.tokenizer
        merged_row_summaries = []

        row_idx = 0
        while row_idx < len(row_summaries):
            current_summary = row_summaries[row_idx]
            while row_idx + 1 < len(row_summaries):
                combined_summary = current_summary + " || " + row_summaries[row_idx + 1]
                if len(tokenizer.encode(combined_summary)) < self.EMBEDDING_MAX_TOKENS:
                    current_summary = combined_summary
                    row_idx += 1
                else:
                    break

            row_idx += 1
            merged_row_summaries.append(current_summary)

        return merged_row_summaries


if __name__ == "__main__":
    fire.Fire(Summarizer)
