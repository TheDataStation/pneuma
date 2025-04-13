import sys
import bm25s

sys.path.append("../..")

from scipy.spatial.distance import cosine
from enum import Enum
from bm25s.tokenization import convert_tokenized_to_string_list
from chromadb.api.models.Collection import Collection
from scipy.spatial.distance import cosine
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline


class RerankingMode(Enum):
    NONE = 0
    COSINE = 1
    LLM = 2
    DIRECT_SCORE = 3


class HybridRetriever:

    def __init__(self, reranker, reranking_mode: RerankingMode) -> None:
        self.reranker = reranker
        self.reranking_mode = reranking_mode

    def _process_nodes_bm25(
        self,
        items,
        all_ids,
        dictionary_id_bm25,
        bm25_retriever: bm25s.BM25,
        query_tokens,
    ):
        results = [node for node in items[0][0]]
        scores = [node for node in items[1][0]]

        extra_results = [
            bm25_retriever.corpus[dictionary_id_bm25[one_id]] for one_id in all_ids
        ]
        extra_scores = [
            bm25_retriever.get_scores(
                convert_tokenized_to_string_list(query_tokens)[0]
            )[dictionary_id_bm25[one_id]]
            for one_id in all_ids
        ]

        results.extend(extra_results)
        scores.extend(extra_scores)

        max_score = max(scores)
        min_score = min(scores)
        processed_nodes = {
            node["metadata"]["table"]: (
                1 if min_score == max_score
                else (scores[i] - min_score) / (max_score - min_score),
                node["text"]
            )
            for i, node in enumerate(results)
        }
        return processed_nodes

    def _process_nodes_vec(
        self, items, missing_ids, collection: Collection, question_embedding
    ):
        extra_information = collection.get_fast(
            ids=missing_ids, limit=len(missing_ids), include=["documents", "embeddings"]
        )
        items["ids"][0].extend(extra_information["ids"])
        items["documents"][0].extend(extra_information["documents"])
        items["distances"][0].extend(
            cosine(question_embedding, extra_information["embeddings"][i])
            for i in range(len(missing_ids))
        )

        scores: list[float] = [1 - dist for dist in items["distances"][0]]
        documents: list[str] = items["documents"][0]
        ids: list[str] = items["ids"][0]

        max_score = max(scores)
        min_score = min(scores)
        processed_nodes = {
            ids[idx]: (
                1 if min_score == max_score
                else (scores[idx] - min_score) / (max_score - min_score),
                documents[idx]
            )
            for idx in range(len(scores))
        }
        return processed_nodes

    def _llm_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        node_tables = [node[0] for node in nodes]

        relevance_prompts = [
            [
                {
                    "role": "user",
                    "content": self._get_relevance_prompt(
                        node[2], 
                        "content" if node[0].split("_SEP_")[1].startswith("contents")
                        else "context", question
                    ),
                }
            ]
            for node in nodes
        ]

        arguments = prompt_pipeline(
            self.reranker,
            relevance_prompts,
            batch_size=2,
            context_length=32768,
            max_new_tokens=2,
            top_p=None,
            temperature=None,
            top_k=None,
        )

        tables_relevance = {
            node_tables[arg_idx]: argument[-1]["content"].lower().startswith("yes")
            for arg_idx, argument in enumerate(arguments)
        }

        new_nodes = [
            (table_name, score, doc)
            for table_name, score, doc in nodes
            if tables_relevance[table_name]
        ] + [
            (table_name, score, doc)
            for table_name, score, doc in nodes
            if not tables_relevance[table_name]
        ]
        return new_nodes

    def _get_relevance_prompt(self, desc: str, desc_type: str, question: str):
        if desc_type == "content":
            return f"""Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
        elif desc_type == "context":
            return f"""Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""

    def _direct_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        names = []
        docs = []
        for node in nodes:
            names.append(node[0])
            docs.append(node[2])
        similarities = self.reranker.compute_score(
            [(question, doc) for doc in docs], normalize=True
        )
        reranked_nodes = sorted(
            zip(names, similarities, docs), key=lambda x: (-x[1], x[0])
        )
        return reranked_nodes
    
    def _cosine_rerank(self, nodes: list[tuple[str, float, str]], question: str):
        # Each node is of the form (name, score, doc)
        names = []
        docs = []
        for node in nodes:
            names.append(node[0])
            docs.append(node[2])

        docs_embeddings = self.reranker.encode(
            docs,
            batch_size=100,
            device="cuda",
        )
        question_embedding = self.reranker.encode(question, device="cuda")
        similarities = [
            1 - cosine(question_embedding, docs_embedding)
            for docs_embedding in docs_embeddings
        ]

        reranked_nodes = sorted(
            zip(names, similarities, docs), key=lambda x: (-x[1], x[0])
        )
        return reranked_nodes

    def retrieve(
        self,
        bm25_retriever: bm25s.BM25,
        vec_retriever,
        bm25_res,
        vec_res,
        k: int,
        question: str,
        alpha=0.5,
        query_tokens=None,
        question_embedding=None,
        dictionary_id_bm25=None,
    ):
        vec_ids = {vec_id for vec_id in vec_res["ids"][0]}
        bm25_ids = {node["metadata"]["table"] for node in bm25_res[0][0]}
        processed_nodes_bm25 = self._process_nodes_bm25(
            bm25_res,
            list(vec_ids - bm25_ids),
            dictionary_id_bm25,
            bm25_retriever,
            query_tokens,
        )
        processed_nodes_vec = self._process_nodes_vec(
            vec_res, list(bm25_ids - vec_ids), vec_retriever, question_embedding
        )

        all_nodes: list[tuple[str, float, str]] = []
        for node_id in sorted(vec_ids | bm25_ids):
            bm25_score_doc = processed_nodes_bm25.get(node_id)
            vec_score_doc = processed_nodes_vec.get(node_id)
            combined_score = alpha * bm25_score_doc[0] + (1 - alpha) * vec_score_doc[0]
            if bm25_score_doc[1] is None:
                doc = vec_score_doc[1]
            else:
                doc = bm25_score_doc[1]
            all_nodes.append((node_id, combined_score, doc))

        sorted_nodes = sorted(all_nodes, key=lambda node: (-node[1], node[0]))[:k]

        if self.reranking_mode == RerankingMode.COSINE:
            sorted_nodes = self._cosine_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.DIRECT_SCORE:
            sorted_nodes = self._direct_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.LLM:
            sorted_nodes = self._llm_rerank(sorted_nodes, question)

        return sorted_nodes
