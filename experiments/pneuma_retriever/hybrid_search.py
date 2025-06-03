import os
import torch
import time
import sys
import chromadb
import bm25s
import Stemmer
import numpy as np

sys.path.append("../..")

from tqdm import tqdm
from transformers import set_seed
from chromadb.api.models.Collection import Collection
from benchmark_generator.context.utils.jsonl import read_jsonl, write_jsonl
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from hybrid_mechanism import HybridRetriever, RerankingMode


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
set_seed(42, deterministic=True)

stemmer = Stemmer.Stemmer("english")
reranker = None
reranking_mode = RerankingMode.NONE

# reranker = initialize_pipeline("../models/qwen", torch.bfloat16)
# reranker.tokenizer.pad_token_id = reranker.model.config.eos_token_id
# reranker.tokenizer.padding_side = "left"
# reranking_mode = RerankingMode.LLM

hybrid_retriever = HybridRetriever(reranker, reranking_mode)
hitrates_data: list[dict[str, str]] = []


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            question_key = "question_from_sql_1"
            benchmark_name = "BC1"
        else:
            question_key = "question"
            benchmark_name = "BC2"
    else:
        if not use_rephrased_questions:
            question_key = "question_bx1"
            benchmark_name = "BX1"
        else:
            question_key = "question_bx2"
            benchmark_name = "BX2"
    return question_key, benchmark_name


def evaluate_benchmark(
    benchmark: list[dict[str, str]],
    benchmark_type: str,
    k: int,
    collection: Collection,
    retriever,
    stemmer,
    dataset: str,
    n=3,
    alpha=0.5,
    use_rephrased_questions=False,
    dictionary_id_bm25=None,
):
    start = time.time()
    hitrate_sum = 0
    wrong_questions = []
    increased_k = k * n
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
        question_embedding = embed_questions[idx]

        query_tokens = bm25s.tokenize(
            questions[idx], stemmer=stemmer, show_progress=False
        )

        results, scores = retriever.retrieve(
            query_tokens, k=increased_k, show_progress=False
        )
        bm25_res = (results, scores)
        vec_res = collection.query(
            query_embeddings=[question_embedding], n_results=increased_k
        )

        all_nodes = hybrid_retriever.retrieve(
            retriever,
            collection,
            bm25_res,
            vec_res,
            increased_k,
            questions[idx],
            alpha,
            query_tokens,
            question_embedding,
            dictionary_id_bm25,
        )
        before = hitrate_sum
        for table, _, _ in all_nodes[:k]:
            table = table.split("_SEP_")[0]
            if table in answer_tables:
                hitrate_sum += 1
                break
        if before == hitrate_sum:
            wrong_questions.append(idx)
        # Checkpoint
        if idx % 100 == 0:
            print(f"Current Hit Rate Sum at index {idx}: {hitrate_sum}")
            print(
                f"Current wrongly answered questions at index {idx}: {wrong_questions}"
            )

    end = time.time()
    print(
        f"Hit Rate (k={k};n={n};alpha={alpha}): {round(hitrate_sum/len(benchmark) * 100, 2)}"
    )
    print(f"Benchmarking Time: {end - start} seconds")
    print(f"Wrongly answered questions: {wrong_questions}")
    hitrates_data.append(
        {
            "dataset": dataset,
            "benchmark_name": benchmark_name,
            "k": k,
            "n": n,
            "alpha": alpha,
            "hitrate": round(hitrate_sum / len(benchmark) * 100, 2),
            "sum": hitrate_sum,
            # "wrong_questions": wrong_questions,
        }
    )
    write_jsonl(hitrates_data, f"hybrid-{reranking_mode}-{k}.jsonl")


def start(
    dataset: str,
    content_benchmark: list[dict[str, str]],
    context_benchmark: list[dict[str, str]],
    alphas: list[int],
    ns: list[int],
    ks=[1],
):
    print(f"Processing {dataset} dataset")
    client = chromadb.PersistentClient(
        f"indices/vector-index-{dataset}-schema_narrations-sample_rows-context"
    )
    collection = client.get_collection("benchmark")
    retriever = bm25s.BM25.load(
        f"indices/fulltext-index-{dataset}-schema_narrations-sample_rows-context",
        load_corpus=True,
    )

    dictionary_id_bm25 = {
        datum["metadata"]["table"]: datum_idx
        for datum_idx, datum in enumerate(retriever.corpus)
    }

    for k in ks:
        for alpha in alphas:
            for n in ns:
                print(f"BC1 (k = {k}) with alpha={alpha} n={n}")
                evaluate_benchmark(
                    benchmark=content_benchmark,
                    benchmark_type="content",
                    k=k,
                    collection=collection,
                    retriever=retriever,
                    stemmer=stemmer,
                    dataset=dataset,
                    n=n,
                    alpha=alpha,
                    use_rephrased_questions=False,
                    dictionary_id_bm25=dictionary_id_bm25,
                )
                print("=" * 50)

                print(f"BC2 (k = {k}) with alpha={alpha} n={n}")
                evaluate_benchmark(
                    benchmark=content_benchmark,
                    benchmark_type="content",
                    k=k,
                    collection=collection,
                    retriever=retriever,
                    stemmer=stemmer,
                    dataset=dataset,
                    n=n,
                    alpha=alpha,
                    use_rephrased_questions=True,
                    dictionary_id_bm25=dictionary_id_bm25,
                )
                print("=" * 50)

                print(f"BX1 (k = {k}) with alpha={alpha} n={n}")
                evaluate_benchmark(
                    benchmark=context_benchmark,
                    benchmark_type="context",
                    k=k,
                    collection=collection,
                    retriever=retriever,
                    stemmer=stemmer,
                    dataset=dataset,
                    n=n,
                    alpha=alpha,
                    use_rephrased_questions=False,
                    dictionary_id_bm25=dictionary_id_bm25,
                )
                print("=" * 50)

                print(f"BX2 (k = {k}) with alpha={alpha} n={n}")
                evaluate_benchmark(
                    benchmark=context_benchmark,
                    benchmark_type="context",
                    k=k,
                    collection=collection,
                    retriever=retriever,
                    stemmer=stemmer,
                    dataset=dataset,
                    n=n,
                    alpha=alpha,
                    use_rephrased_questions=True,
                    dictionary_id_bm25=dictionary_id_bm25,
                )
                print("=" * 50)


if __name__ == "__main__":
    # Adjust
    ns = [5]
    alphas = [0.5]
    ks = [1]

    dataset = "chembl"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(
        dataset,
        content_benchmark,
        context_benchmark,
        alphas,
        ns,
        ks,
    )

    dataset = "adventure"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(
        dataset,
        content_benchmark,
        context_benchmark,
        alphas,
        ns,
        ks,
    )

    dataset = "public"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(
        dataset,
        content_benchmark,
        context_benchmark,
        alphas,
        ns,
        ks,
    )

    dataset = "chicago"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(
        dataset,
        content_benchmark,
        context_benchmark,
        alphas,
        ns,
        ks,
    )

    dataset = "fetaqa"
    content_benchmark = read_jsonl(
        "../../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset}/bx_{dataset}.jsonl"
    )
    start(
        dataset,
        content_benchmark,
        context_benchmark,
        alphas,
        ns,
        ks,
    )
