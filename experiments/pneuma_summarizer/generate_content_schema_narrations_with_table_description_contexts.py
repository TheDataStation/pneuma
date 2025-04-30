# NOT PART OF EXPERIMENTS OF THE PAPER
import os
import gc

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import torch
import argparse
import pandas as pd

sys.path.append("../..")

from tqdm import tqdm
from collections import defaultdict
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import (
    prompt_pipeline,
    prompt_pipeline_robust,
)
from benchmark_generator.context.utils.jsonl import write_jsonl, read_jsonl


def get_col_description_prompt(columns: str, column: str, table_description: str):
    return f"""A table, which represents {table_description}, has the following columns:
/*
{columns}
*/
Describe very briefly what the {column} column represents. If not possible, simply state "No description.\""""


def str_to_bool(value: str) -> bool:
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError("Invalid boolean value")


def get_special_indices(texts: list[str], batch_size: int):
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


def is_fit_in_memory(conversations, batch_size: int, hallucinate: bool):
    special_indices = get_special_indices(conversations, batch_size)
    adjusted_conversations = [conversations[i] for i in special_indices]

    conv_low_idx = len(adjusted_conversations) // 2 - batch_size // 2
    conv_high_idx = conv_low_idx + batch_size

    if hallucinate:
        output = prompt_pipeline(
        pipe,
        adjusted_conversations[conv_low_idx:conv_high_idx],
        batch_size=batch_size,
        context_length=32768,
        max_new_tokens=1,
        do_sample=True,
        temperature=1.5,
    )
    else:
        output = prompt_pipeline(
            pipe,
            adjusted_conversations[conv_low_idx:conv_high_idx],
            batch_size=batch_size,
            context_length=32768,
            max_new_tokens=1,
            temperature=None,
            top_k=None,
            top_p=None,
        )

    torch.cuda.empty_cache()
    gc.collect()

    if output[0][0]["content"] == "":
        del output
        return False
    else:
        del output
        return True


def get_optimal_batch_size(conversations, hallucinate: bool):
    return 30
    print("Looking for an optimal batch size")
    max_batch_size = (
        50  # Change to a higher value if you have more capacity to explore batch size
    )
    min_batch_size = 1
    while min_batch_size < max_batch_size:
        mid_batch_size = (min_batch_size + max_batch_size) // 2
        print(f"Current mid batch size: {mid_batch_size}")
        if is_fit_in_memory(conversations, mid_batch_size, hallucinate):
            min_batch_size = mid_batch_size + 1
        else:
            max_batch_size = mid_batch_size - 1
    optimal_batch_size = min_batch_size
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def parse_tables(tables: list[str], tables_path: str, contexts: dict[str,str]):
    conversations: list[str] = []
    conv_tables: list[str] = []
    conv_cols: list[str] = []
    for table in tqdm(tables):
        try:
            df = pd.read_csv(f"{tables_path}/{table}.csv", nrows=0)
        except pd.errors.EmptyDataError:
            continue
        table_context = [i['context'] for i in contexts if i['table'] == table][0]
        cols = df.columns
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col, table_context)
            conversations.append([{"role": "user", "content": prompt}])
            conv_tables.append(table)
            conv_cols.append(col)
    return conversations, conv_tables, conv_cols


def generate_schema_narration_summaries(
    tables_path: str, summaries_name: str, hallucinate: bool, model_name: str, contexts: dict[str,str]
):
    tables = sorted([file[:-4] for file in os.listdir(tables_path)])
    summaries: list[dict[str, str]] = []

    conversations, conv_tables, conv_cols = parse_tables(tables, tables_path, contexts)
    optimal_batch_size = get_optimal_batch_size(conversations, hallucinate)
    sorted_indices = get_special_indices(conversations, optimal_batch_size)

    conversations = [conversations[i] for i in sorted_indices]
    conv_tables = [conv_tables[i] for i in sorted_indices]
    conv_cols = [conv_cols[i] for i in sorted_indices]

    SCHEMA_NARRATIONS_PATH = "summaries/schema_narrations"
    if hallucinate:
        if model_name == "qwen":
            SCHEMA_NARRATIONS_PATH = "summaries/temperature-1.5-instruct"
        else:
            SCHEMA_NARRATIONS_PATH = "summaries/temperature-1.5-none"

    if len(conversations) > 0:
        outputs = []
        max_batch_size = optimal_batch_size
        same_batch_size_counter = 0
        for i in tqdm(range(0, len(conversations), max_batch_size)):
            if hallucinate:
                llm_output = prompt_pipeline_robust(
                    pipe,
                    conversations[i : i + max_batch_size],
                    batch_size=optimal_batch_size,
                    context_length=32768,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=1.5,
                )
            else:
                llm_output = prompt_pipeline_robust(
                    pipe,
                    conversations[i : i + max_batch_size],
                    batch_size=optimal_batch_size,
                    context_length=32768,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=None,
                    top_k=None,
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
                f"{conv_cols[output_idx]}: {output[-1]["content"]}"
            ]

        for table in tables:
            summaries.append(
                {
                    "id": f"{table}_SEP_contents_SEP_schema",
                    "table": table,
                    "summary": " | ".join(col_narrations[table]),
                }
            )
            write_jsonl(summaries, f"{SCHEMA_NARRATIONS_PATH}/{summaries_name}.jsonl")
    summaries = sorted(summaries, key=lambda x: x["table"])
    write_jsonl(summaries, f"{SCHEMA_NARRATIONS_PATH}/{summaries_name}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program generates SchemaNarrations summaries, which is\
                    basically descriptions of schemas.",
        epilog="Alternatively, you may download the generated summaries from\
                the `summaries` directory.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    parser.add_argument("-m", "--model", default="qwen")
    parser.add_argument(
        "-hal",
        "--hallucinate",
        type=str_to_bool,
        default=False,
        choices=[True, False],
    )
    dataset: str = parser.parse_args().dataset
    hallucinate: bool = parser.parse_args().hallucinate
    model_name: str = parser.parse_args().model

    pipe = initialize_pipeline(f"../models/{model_name}", torch.bfloat16, context_length=32768)

    # Specific settings for batching
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"

    with open("constants.json") as file:
        constants: dict[str, any] = json.load(file)

    TABLES_SRC: str = constants["data_src"] + "tables/"
    TABLES: dict[str, str] = constants["tables"]

    # Hardcoded ctx
    contexts = read_jsonl('../../../buysite_data/context.jsonl')

    if dataset == "all":
        for table_info in TABLES.items():
            summaries_name, table_name = table_info
            tables_path = TABLES_SRC + table_name
            generate_schema_narration_summaries(tables_path, summaries_name, hallucinate, model_name, contexts)
    else:
        try:
            table_name = TABLES[dataset]
            tables_path = TABLES_SRC + table_name
            generate_schema_narration_summaries(tables_path, dataset, hallucinate, model_name, contexts)
        except KeyError:
            print(
                f"Dataset {dataset} not found! Please define the path in `constants.json`."
            )
