import argparse
import sys

sys.path.append("..")

import bm25s
import time
import Stemmer
from commons import DATASETS, str_to_bool, get_documents


stemmer = Stemmer.Stemmer("english")


def indexing_full_text(
    stemmer,
    schema_contents: list[dict[str, str]],
    row_contents: list[dict[str, str]],
    schema_content_type: str,
    row_content_type: str,
    contexts: list[dict[str, str]],
):
    start = time.time()
    corpus_json = []

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
                corpus_json.append(
                    {
                        "text": schema_content["summary"],
                        "metadata": {
                            "table": f"{table}_SEP_contents_SEP_schema-{content_idx}"
                        },
                    }
                )

        if row_contents is not None:
            table_row_contents = [
                content for content in row_contents if content["table"] == table
            ]

            for content_idx, row_content in enumerate(table_row_contents):
                corpus_json.append(
                    {
                        "text": row_content["summary"],
                        "metadata": {
                            "table": f"{table}_SEP_contents_SEP_row-{content_idx}"
                        },
                    }
                )

        if contexts is not None:
            table_contexts = [
                context for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(table_contexts):
                corpus_json.append(
                    {
                        "text": context["context"],
                        "metadata": {"table": f"{table}_SEP_contexts-{context_idx}"},
                    }
                )

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(
        f"indices/fulltext-index-{dataset}{f'-{schema_content_type}' if schema_content_type != "none" else ''}{f'-{row_content_type}' if row_content_type != "none" else ''}{'-context' if include_contexts else ''}"
    )
    end = time.time()
    print(f"Indexing time of dataset {dataset}: {end-start} seconds")


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
            indexing_full_text(
                stemmer,
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
        indexing_full_text(
            stemmer,
            schema_contents,
            row_contents,
            schema_content_type,
            row_content_type,
            contexts,
        )
