import json
import logging
import os
import sys
import time
from pathlib import Path

import bm25s
import chromadb
import duckdb
import fire
import pandas as pd
import Stemmer
import tiktoken
from chromadb.db.base import UniqueConstraintError
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_config import configure_logging
from utils.prompting_interface import prompt_openai_embed
from utils.response import Response, ResponseStatus
from utils.storage_config import get_storage_path
from utils.summary_types import SummaryType

configure_logging()
logger = logging.getLogger("IndexGenerator")


class IndexGenerator:
    def __init__(
        self,
        embed_model,
        db_path: str = os.path.join(get_storage_path(), "storage.db"),
        index_path: str = None,
    ):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)
        self.embedding_model = embed_model
        self.stemmer = Stemmer.Stemmer("english")

        if index_path is None:
            index_path = os.path.join(os.path.dirname(db_path), "indexes")
        self.index_path = index_path
        self.vector_index_path = os.path.join(index_path, "vector")
        self.keyword_index_path = os.path.join(index_path, "keyword")
        self.chroma_client = chromadb.PersistentClient(self.vector_index_path)

        if isinstance(self.embedding_model, OpenAI):
            self.EMBEDDING_MAX_TOKENS = 8191
        else:
            self.EMBEDDING_MAX_TOKENS = 512

    def generate_index(self, index_name: str, table_ids: list | tuple = None) -> str:
        if table_ids is None:
            logger.info("No table ids provided. Generating index for all tables...")
            table_ids = [
                entry[0]
                for entry in self.connection.sql(
                    "SELECT id FROM table_status"
                ).fetchall()
            ]
        elif isinstance(table_ids, str):
            table_ids = (table_ids,)

        logger.info("Generating index for %d tables...", len(table_ids))

        ### GENERATING AND INSERTING TABLES TO VECTOR INDEX ###
        start_time = time.time()
        vector_index_response = self.__generate_vector_index(index_name)
        end_time = time.time()
        vector_index_generation_time = end_time - start_time
        if vector_index_response.status == ResponseStatus.ERROR:
            return vector_index_response.to_json()

        vector_index_id = vector_index_response.data["index_id"]
        chroma_collection = vector_index_response.data["collection"]

        logger.info(vector_index_response.message)
        vector_insert_response = self.__insert_tables_to_vector_index(
            vector_index_id, table_ids, chroma_collection
        )

        if vector_insert_response.status == ResponseStatus.ERROR:
            self.chroma_client.delete_collection(index_name)
            return vector_insert_response.to_json()

        logger.info(vector_insert_response.message)

        ### GENERATING AND INSERTING TABLES TO KEYWORD INDEX ###
        start_time = time.time()
        keyword_index_response = self.__generate_keyword_index(index_name)
        end_time = time.time()
        keyword_index_generation_time = end_time - start_time
        if keyword_index_response.status == ResponseStatus.ERROR:
            self.chroma_client.delete_collection(index_name)
            return keyword_index_response.to_json()

        keyword_index_id = keyword_index_response.data["index_id"]
        retriever = keyword_index_response.data["retriever"]

        logger.info(keyword_index_response.message)
        keyword_insert_response = self.__insert_tables_to_keyword_index(
            keyword_index_id, table_ids, retriever
        )

        if keyword_insert_response.status == ResponseStatus.ERROR:
            self.chroma_client.delete_collection(index_name)
            return keyword_insert_response.to_json()

        logger.info(keyword_insert_response.message)

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Vector and keyword index named {index_name} with id {vector_index_id}"
            f" and {keyword_index_id} has been created with {len(table_ids)} tables.",
            data={
                "table_ids": table_ids,
                "vector_index_id": vector_index_id,
                "keyword_index_id": keyword_index_id,
                "vector_index_generation_time": vector_index_generation_time,
                "keyword_index_generation_time": keyword_index_generation_time,
            },
        ).to_json()

    def __generate_vector_index(self, index_name: str) -> Response:
        try:
            chroma_collection = self.chroma_client.create_collection(
                name=index_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:random_seed": 42,
                    "hnsw:M": 48,
                },
            )
        except UniqueConstraintError:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Index named {index_name} already exists.",
            )

        # If we decide to use DuckDB's vector store, we don't need to store
        # this data.
        index_id = self.connection.sql(
            f"""INSERT INTO indexes (name, location)
            VALUES ('{index_name}', '{self.vector_index_path}')
            RETURNING id"""
        ).fetchone()[0]

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Vector index named {index_name} with id {index_id} has been created.",
            data={
                "index_id": index_id,
                "collection": chroma_collection,
            },
        )

    def __insert_tables_to_vector_index(
        self,
        index_id: int,
        table_ids: list | tuple,
        chroma_collection: chromadb.Collection,
    ) -> Response:
        documents = []
        ids = []

        for table_id in table_ids:
            logger.info("Processing table %s...", table_id)

            narration_summaries = self.__get_table_summaries(
                table_id, SummaryType.NARRATION
            )
            row_summaries = self.__get_table_summaries(
                table_id, SummaryType.ROW_SUMMARY
            )

            for idx, narration_summary in enumerate(narration_summaries):
                documents.append(json.loads(narration_summary[1])["payload"])
                ids.append(f"{table_id}_SEP_contents_SEP_schema-{idx}")

            for idx, row_summary in enumerate(row_summaries):
                documents.append(json.loads(row_summary[1])["payload"])
                ids.append(f"{table_id}_SEP_contents_SEP_row-{idx}")

            contexts = self.__get_table_contexts(table_id)
            if not contexts:
                continue
            contexts = self.__merge_contexts(contexts)
            for context_idx, context in enumerate(contexts):
                documents.append(context)
                ids.append(f"{table_id}_SEP_contexts-{context_idx}")

        if len(documents) == 0:
            return Response(
                status=ResponseStatus.ERROR,
                message="No context and summary entries found for the given table ids.",
            )

        for i in tqdm(range(0, len(documents), 30000)):
            if isinstance(self.embedding_model, OpenAI):
                embeddings = prompt_openai_embed(
                    self.embedding_model, documents[i : i + 30000],
                )
                chroma_collection.add(
                    embeddings=embeddings,
                    documents=documents[i : i + 30000],
                    ids=ids[i : i + 30000],
                )
            else:
                embeddings = self.embedding_model.encode(
                    documents[i : i + 30000],
                    batch_size=10,
                    show_progress_bar=True,
                    device="cuda",
                )
                chroma_collection.add(
                    embeddings=[embed.tolist() for embed in embeddings],
                    documents=documents[i : i + 30000],
                    ids=ids[i : i + 30000],
                )

        insert_df = pd.DataFrame.from_dict(
            {
                "index_id": [index_id] * len(table_ids),
                "table_id": table_ids,
            }
        )

        # So we know which tables are included in this index.
        self.connection.sql(
            """INSERT INTO index_table_mappings (index_id, table_id)
            SELECT * FROM insert_df""",
        )

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{len(table_ids)} Tables have been inserted to index with id {index_id}.",
        )

    def __generate_keyword_index(self, index_name):
        retriever = bm25s.BM25(corpus=[])
        corpus_tokens = bm25s.tokenize([])
        retriever.index(corpus_tokens)
        retriever.save(os.path.join(self.keyword_index_path, index_name), corpus=[])

        index_id = self.connection.sql(
            f"""INSERT INTO indexes (name, location)
            VALUES ('{index_name}', '{self.keyword_index_path}')
            RETURNING id"""
        ).fetchone()[0]

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Keyword index named {index_name} with id {index_id} has been created.",
            data={
                "index_id": index_id,
                "retriever": retriever,
            },
        )

    def __insert_tables_to_keyword_index(
        self, index_id: int, table_ids: list | tuple, retriever: bm25s.BM25
    ):
        index_name = self.connection.sql(
            f"SELECT name FROM indexes WHERE id = {index_id}"
        ).fetchone()[0]

        corpus_json = []
        for table_id in table_ids:
            logger.info("Processing table %s...", table_id)

            narration_summaries = self.__get_table_summaries(
                table_id, SummaryType.NARRATION
            )

            row_summaries = self.__get_table_summaries(
                table_id, SummaryType.ROW_SUMMARY
            )

            for idx, narration_summary in enumerate(narration_summaries):
                content = json.loads(narration_summary[1])["payload"]
                corpus_json.append(
                    {
                        "text": content,
                        "metadata": {
                            "table": f"{table_id}_SEP_contents_SEP_schema-{idx}"
                        },
                    }
                )

            for idx, row_summary in enumerate(row_summaries):
                content = json.loads(row_summary[1])["payload"]
                corpus_json.append(
                    {
                        "text": content,
                        "metadata": {"table": f"{table_id}_SEP_contents_SEP_row-{idx}"},
                    }
                )

            contexts = self.__get_table_contexts(table_id)
            if not contexts:
                continue
            contexts = self.__merge_contexts(contexts)
            for context_idx, context in enumerate(contexts):
                corpus_json.append(
                    {
                        "text": context,
                        "metadata": {"table": f"{table_id}_SEP_contexts-{context_idx}"},
                    }
                )

        corpus_text = [doc["text"] for doc in corpus_json]
        corpus_tokens = bm25s.tokenize(
            corpus_text, stopwords="en", stemmer=self.stemmer, show_progress=False
        )

        retriever.corpus = retriever.corpus + corpus_json
        retriever.index(corpus_tokens, show_progress=False)

        retriever.save(
            os.path.join(self.keyword_index_path, index_name), corpus=retriever.corpus
        )

        insert_df = pd.DataFrame.from_dict(
            {
                "index_id": [index_id] * len(table_ids),
                "table_id": table_ids,
            }
        )

        # So we know which tables are included in this index.
        self.connection.sql(
            """INSERT INTO index_table_mappings (index_id, table_id)
            SELECT * FROM insert_df""",
        )

        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{len(table_ids)} Tables have been inserted to index with id {index_id}.",
        )

    def __get_table_contexts(self, table_id: str) -> list[tuple[str, str]]:
        table_id = table_id.replace("'", "''")
        return self.connection.sql(
            f"""SELECT id, context FROM table_contexts
            WHERE table_id='{table_id}'"""
        ).fetchall()

    def __merge_contexts(self, contexts: list[tuple[str, str]]) -> list[str]:
        if isinstance(self.embedding_model, OpenAI):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = self.embedding_model.tokenizer
        table_contexts = [json.loads(context[1])["payload"] for context in contexts]
        processed_contexts = []

        context_idx = 0
        while context_idx < len(table_contexts):
            current_context = table_contexts[context_idx]
            while (context_idx + 1) < len(table_contexts):
                combined_context = (
                    current_context + " | " + table_contexts[context_idx + 1]
                )
                if len(tokenizer.encode(combined_context)) < self.EMBEDDING_MAX_TOKENS:
                    current_context = combined_context
                    context_idx += 1
                else:
                    break

            context_idx += 1
            processed_contexts.append(current_context)

        return processed_contexts

    def __get_table_summaries(
        self, table_id: str, summary_type: SummaryType
    ) -> list[tuple[str, str]]:
        table_id = table_id.replace("'", "''")
        return self.connection.sql(
            f"""SELECT id, summary FROM table_summaries
            WHERE table_id='{table_id}' AND summary_type='{summary_type}'"""
        ).fetchall()


if __name__ == "__main__":
    fire.Fire(IndexGenerator)
