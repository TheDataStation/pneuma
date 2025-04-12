import datetime
import json
import sys
import time

import bm25s
import Stemmer

sys.path.append("../..")

from tqdm import tqdm

from benchmark_generator.context.utils.jsonl import read_jsonl, write_jsonl

hitrates_data: list[dict[str, str]] = []
stemmer = Stemmer.Stemmer("english")


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            question_key = "question_from_sql_1"
            benchmark_name = "bc1"
        else:
            question_key = "question"
            benchmark_name = "bc2"
    else:
        if not use_rephrased_questions:
            question_key = "question_bx1"
            benchmark_name = "bx1"
        else:
            question_key = "question_bx2"
            benchmark_name = "bx2"
    return (question_key, benchmark_name)


def evaluate_benchmark(
    benchmark: list[dict[str, str]],
    benchmark_type: str,
    k: int,
    retriever,
    stemmer,
    dataset: str,
    use_rephrased_questions=False,
):
    hitrate_sum = 0
    question_key, benchmark_name = get_question_key(
        benchmark_type, use_rephrased_questions
    )

    questions = []
    for data in benchmark:
        questions.append(data[question_key])

    q_count = 0
    for idx, datum in enumerate(tqdm(benchmark)):
        if q_count >= 50:
            break
        answer_tables = datum["answer_tables"]

        query_tokens = bm25s.tokenize(
            questions[idx], stemmer=stemmer, show_progress=False
        )
        results, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
        for result in results[0]:
            table = result["metadata"]["table"].split("_SEP_")[0]
            if table in answer_tables:
                hitrate_sum += 1
                break
        q_count += 1

    print(f"Hit Rate: {round(hitrate_sum/len(benchmark) * 100, 2)}")
    hitrates_data.append(
        {
            "dataset": dataset,
            "benchmark": benchmark_name,
            "k": k,
            "hitrate": round(hitrate_sum / len(benchmark) * 100, 2),
        }
    )


def start(
    dataset: str,
    content_benchmark: list[dict[str, str]],
    k: int = 1,
    table_count: int = 10330,
):
    retriever = bm25s.BM25.load(
        f"indices/keyword-index-{dataset}-{table_count}", load_corpus=True
    )

    print(f"BC1 with k={k}")
    start = time.time()
    evaluate_benchmark(content_benchmark, "content", k, retriever, stemmer, dataset)
    end = time.time()
    bc1_time = end - start
    print(f"Indexing time for {dataset}: {bc1_time} seconds")

    print(f"BC2 with k={k}")
    start = time.time()
    evaluate_benchmark(
        content_benchmark, "content", k, retriever, stemmer, dataset, True
    )
    end = time.time()
    bc2_time = end - start
    print(f"Indexing time for {dataset}: {bc2_time} seconds")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = "benchmark_results"
    with open(
        f"{out_path}/benchmark-{dataset}-{timestamp}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json_results = {
            "dataset": dataset,
            "timestamp": timestamp,
            "table_count": table_count,
            "evaluation_time": bc1_time + bc2_time,
            "bc1_time": bc1_time,
            "bc2_time": bc2_time,
        }
        json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    table_count = 1250
    dataset = "fetaqa"
    benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    for i in range(10):
        start(dataset, benchmark, table_count=table_count)
        time.sleep(1.5)
