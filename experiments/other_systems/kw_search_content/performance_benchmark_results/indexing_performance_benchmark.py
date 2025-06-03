import datetime
import json
import os
import sys
import time

import bm25s
import Stemmer

sys.path.append("../..")


def get_table_contents(table_path: str, table_name: str):
    with open(table_path, "r", encoding="utf-8") as file:
        table_content = file.read()
        document = {
            "text": table_content,
            "metadata": {"table": f"{table_name}_SEP_contents"},
        }
        return document


def index_dataset(dataset_path: str, dataset_name: str, read_limit: int = 10330):
    stemmer = Stemmer.Stemmer("english")
    corpus_json: list[dict] = []

    read_count = 0
    for table in sorted(os.listdir(dataset_path)):
        if read_count >= read_limit:
            break
        table_contents = get_table_contents(f"{dataset_path}/{table}", table[:-4])

        corpus_json.append(table_contents)
        read_count += 1
    print(f"Read {read_count} tables from {dataset_name}")

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(f"indices/keyword-index-{dataset_name}-{read_count}")


if __name__ == "__main__":
    read_limit = 625
    dataset_name = "fetaqa"
    dataset_path = "../../data_src/tables/pneuma_fetaqa"
    start = time.time()
    index_dataset(dataset_path, dataset_name, read_limit=read_limit)
    end = time.time()

    print(f"Indexing time for {dataset_name}: {end-start} seconds")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = "benchmark_results"
    with open(
        f"{out_path}/benchmark-{dataset_name}-{timestamp}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json_results = {
            "dataset": dataset_name,
            "timestamp": timestamp,
            "indexing_time": end - start,
            "read_limit": read_limit,
        }
        json.dump(json_results, f, indent=4)
