import sys
import time
import chromadb
import numpy as np

sys.path.append("..")

from tqdm import tqdm
from chromadb.api.models.Collection import Collection
from benchmark_generator.context.utils.jsonl import read_jsonl, write_jsonl

hitrates_data: list[dict[str, str]] = []


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
    collection: Collection,
    dataset: str,
    use_rephrased_questions=False,
):
    start = time.time()
    hitrate_sum = 0
    wrong_questions = []
    question_key, benchmark_name = get_question_key(
        benchmark_type, use_rephrased_questions
    )

    questions = []
    for data in benchmark:
        questions.append(data[question_key])
    embed_questions = np.loadtxt(
        f"embeddings/embed-{dataset}-questions-{benchmark_type}-{use_rephrased_questions}.txt"
    )
    embed_questions = [embed.tolist() for embed in embed_questions]

    for idx, datum in enumerate(tqdm(benchmark)):
        answer_tables = datum["answer_tables"]
        vec_res = collection.query(query_embeddings=[embed_questions[idx]], n_results=k)
        before = hitrate_sum
        for res in vec_res["ids"][0]:
            table = res.split("_SEP_")[0]
            if table in answer_tables:
                hitrate_sum += 1
                break
        if before == hitrate_sum:
            wrong_questions.append(idx)

    end = time.time()

    print(f"Hit Rate: {round(hitrate_sum/len(benchmark) * 100, 2)}")
    print(f"Benchmarking Time: {end - start} seconds")
    print(f"Wrongly answered questions: {wrong_questions}")

    hitrates_data.append(
        {
            "dataset": dataset,
            "benchmark": benchmark_name,
            "k": k,
            "hitrate": round(hitrate_sum / len(benchmark) * 100, 2),
        }
    )
    write_jsonl(hitrates_data, "vector.jsonl")


def start(
    dataset: str,
    content_benchmark: list[dict[str, str]],
    context_benchmark: list[dict[str, str]],
    ks: list[int],
):
    start = time.time()
    client = chromadb.PersistentClient(f"indices/index-{dataset}-pneuma-summarizer")
    collection = client.get_collection("benchmark")
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    for k in ks:
        print(f"BC1 with k={k}")
        evaluate_benchmark(content_benchmark, "content", k, collection, dataset)

    for k in ks:
        print(f"BC2 with k={k}")
        evaluate_benchmark(content_benchmark, "content", k, collection, dataset, True)

    for k in ks:
        print(f"BX1 with k={k}")
        evaluate_benchmark(context_benchmark, "context", k, collection, dataset)

    for k in ks:
        print(f"BX2 with k={k}")
        evaluate_benchmark(context_benchmark, "context", k, collection, dataset, True)


if __name__ == "__main__":
    ks = [1, 10, 50]

    dataset = "chembl"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(dataset, content_benchmark, context_benchmark, ks)

    dataset = "adventure"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(dataset, content_benchmark, context_benchmark, ks)

    dataset = "public"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(dataset, content_benchmark, context_benchmark, ks)

    dataset = "chicago"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(dataset, content_benchmark, context_benchmark, ks)

    dataset = "fetaqa"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(dataset, content_benchmark, context_benchmark, ks)
