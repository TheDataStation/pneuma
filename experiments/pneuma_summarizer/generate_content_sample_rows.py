import os
import sys
import math
import json
import argparse
import pandas as pd

sys.path.append("../..")

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl


def generate_sample_rows_summaries(table_path: str, summary_path: str):
    content_summaries = []
    tables = sorted(os.listdir(table_path))
    for table_idx, table in enumerate(tqdm(tables)):
        try:
            df = pd.read_csv(f"{table_path}/{table}", on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            continue
        sample_size = math.ceil(min(len(df), 5))

        selected_df = df.sample(n=sample_size, random_state=table_idx).reset_index(
            drop=True
        )
        for df_idx, row in selected_df.iterrows():
            formatted_row = " | ".join([f"{col}: {val}" for col, val in row.items()])
            content_summaries.append(
                {
                    "id": f"{table[:-4]}_SEP_contents_SEP_row-{df_idx}",
                    "table": table[:-4],
                    "summary": formatted_row,
                }
            )
    SAMPLE_ROWS_PATH = "summaries/sample_rows"
    try:
        write_jsonl(content_summaries, f"{SAMPLE_ROWS_PATH}/{summary_path}.jsonl")
    except FileNotFoundError:
        os.mkdir(SAMPLE_ROWS_PATH)
        write_jsonl(content_summaries, f"{SAMPLE_ROWS_PATH}/{summary_path}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program generates SampleRows summaries, which is\
                    basically all rows of the tables.",
        epilog="Alternatively, you may download the generated summaries from\
                the `summaries` directory.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    dataset = parser.parse_args().dataset

    with open("constants.json") as file:
        constants: dict[str, any] = json.load(file)
    
    TABLES_SRC: str = constants["data_src"] + "tables/"
    TABLES: dict[str, str] = constants["tables"]

    if dataset == "all":
        for table_info in TABLES.items():
            summaries_name, table_name = table_info
            tables_path = TABLES_SRC + table_name
            generate_sample_rows_summaries(tables_path, summaries_name)
    else:
        try:
            table_name = TABLES[dataset]
            tables_path = TABLES_SRC + table_name
            generate_sample_rows_summaries(tables_path, dataset)
        except KeyError:
            print(
                f"Dataset {dataset} not found! Please define the path in `constants.json`."
            )
