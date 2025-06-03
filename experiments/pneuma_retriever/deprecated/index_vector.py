import argparse
import os
import time
import sys
import chromadb

sys.path.append("..")

from transformers import set_seed
from chromadb.api.client import Client
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from commons import DATASETS, str_to_bool, get_documents


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


embedding_model = SentenceTransformer("../models/bge-base", local_files_only=True)


def indexing_vector(
    client: Client,
    embedding_model: SentenceTransformer,
    schema_contents: list[dict[str, str]],
    row_contents: list[dict[str, str]],
    contexts: list[dict[str, str]] = None,
    collection_name="benchmark",
    reindex=False,
):
    if not reindex:
        try:
            collection = client.get_collection(collection_name)
            return collection
        except:
            pass
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:random_seed": 42,
            "hnsw:M": 48,
        },
    )

    documents = []
    ids = []

    if row_contents is not None:
        tables = sorted({content["table"] for content in row_contents})
    else:
        tables = sorted({content["table"] for content in schema_contents})
    for table in tables:
        if schema_contents is not None:
            table_schema_contents = [
                content for content in schema_contents if content["table"] == table
            ]

            for content_idx, schema_content in enumerate(table_schema_contents):
                documents.append(schema_content["summary"])
                ids.append(f"{table}_SEP_contents_SEP_schema-{content_idx}")

        if row_contents is not None:
            table_row_contents = [
                content for content in row_contents if content["table"] == table
            ]

            for content_idx, row_content in enumerate(table_row_contents):
                documents.append(row_content["summary"])
                ids.append(f"{table}_SEP_contents_SEP_row-{content_idx}")

        if contexts is not None:
            table_contexts = [
                context for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(table_contexts):
                documents.append(context["context"])
                ids.append(f"{table}_SEP_contexts-{context_idx}")

    for i in range(0, len(documents), 30000):
        embeddings = embedding_model.encode(
            documents[i : i + 30000],
            batch_size=100,
            show_progress_bar=True,
            device="cuda",
        )

        collection.add(
            embeddings=[embed.tolist() for embed in embeddings],
            documents=documents[i : i + 30000],
            ids=ids[i : i + 30000],
        )
    return collection


def start_indexing(
    dataset, schema_contents, row_contents, schema_content_type, row_content_type, contexts
):
    print(f"Indexing dataset: {dataset}")
    start = time.time()
    client = chromadb.PersistentClient(f"indices/vector-index-{dataset}{f'-{schema_content_type}' if schema_content_type != "none" else ''}{f'-{row_content_type}' if row_content_type != "none" else ''}{'-context' if include_contexts else ''}")
    indexing_vector(
        client, embedding_model, schema_contents, row_contents, contexts
    )
    end = time.time()
    print(f"Indexing time: {end-start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program indexes content and context documents.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="all",
        choices=["chembl", "adventure", "public", "chicago", "fetaqa", "bird"],
    )
    parser.add_argument(
        "-sctn",
        "--schema-content-type",
        default="schema_narrations",
        choices=["schema_narrations", "schema_concat", "none"],
    )
    parser.add_argument(
        "-rctn",
        "--row-content-type",
        default="sample_rows",
        choices=["sample_rows", "dbreader", "none"],
    )
    parser.add_argument(
        "-ctx",
        "--include-contexts",
        type=str_to_bool,
        default=False,
        choices=[True, False],
    )
    dataset: str = parser.parse_args().dataset
    schema_content_type: str = parser.parse_args().schema_content_type
    row_content_type: str = parser.parse_args().row_content_type
    include_contexts: bool = parser.parse_args().include_contexts

    if schema_content_type == "none" and row_content_type == "none":
        raise ValueError(
            "At least one of schema and row content types must not be none."
        )

    if dataset == "all":
        for dataset in DATASETS.keys():
            schema_contents, row_contents, contexts = get_documents(
                dataset, schema_content_type, row_content_type, include_contexts
            )
            start_indexing(
                dataset,
                schema_contents,
                row_contents,
                schema_content_type,
                row_content_type,
                contexts,
            )
    else:
        schema_contents, row_contents, contexts = get_documents(
            dataset, schema_content_type, row_content_type, include_contexts
        )
        start_indexing(
            dataset,
            schema_contents,
            row_contents,
            schema_content_type,
            row_content_type,
            contexts,
        )
