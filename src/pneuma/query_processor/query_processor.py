import logging
import os

import bm25s
import fire
import Stemmer
from bm25s.tokenization import Tokenized, convert_tokenized_to_string_list
from chromadb_deterministic import PersistentClient
from chromadb_deterministic.api.models.Collection import Collection
from chromadb_deterministic.api.types import QueryResult
from numpy import ndarray
from openai import OpenAI
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import TextGenerationPipeline

from pneuma.utils.logging_config import configure_logging
from pneuma.utils.prompting_interface import (
    prompt_openai_embed,
    prompt_openai_llm,
    prompt_pipeline,
)
from pneuma.utils.response import Response, ResponseStatus
from pneuma.utils.storage_config import get_storage_path

configure_logging()
logger = logging.getLogger("Registrar")


class QueryProcessor:
    """
    Processes queries against generated hybrid indexes.

    This class provides a method to retrieve tables from hybrid indexes,
    helping people find the relevant tables for their tasks.

    ## Attributes
    - **pipe** (`OpenAI | TextGenerationPipeline`): The LLM pipeline for inference.
    - **embedding_model** (`OpenAI | SentenceTransformer`): The model used for
    text embeddings.
    - **stemmer** (`Stemmer`): A stemming tool used for text normalization.
    - **db_path** (`str`): Path to the database file for retrieving content
    summaries & context.
    - **index_path** (`str`): Path to the directory where indexes are stored.
    - **vector_index_path** (`str`): Path for vector-based indexing.
    - **fulltext_index_path** (`str`): Path for full-text search indexing.
    """

    def __init__(
        self,
        llm: OpenAI | TextGenerationPipeline,
        embed_model: OpenAI | SentenceTransformer,
        db_path: str = None,
        index_path: str = None,
    ):
        self.pipe = llm
        self.embedding_model = embed_model
        self.stemmer = Stemmer.Stemmer("english")

        if db_path is None:
            db_path = os.path.join(get_storage_path(), "storage.db")
        if index_path is None:
            index_path = os.path.join(os.path.dirname(db_path), "indexes")

        self.vector_index_path = os.path.join(index_path, "vector")
        self.fulltext_index_path = os.path.join(index_path, "fulltext")

    def query(
        self,
        index_name: str,
        queries: str | list[str],
        k: int = 1,
        n: int = 5,
        alpha: float = 0.5,
    ) -> str:
        """
        Retrieves tables for the given `queries` against the index `index_name`.

        ## Args
        - **index_name** (`str`): The name of the index to be retrieved against.
        - **queries** (`str | list[str]`): The query of list of queries to be executed.
        - **k** (`int`): The number of documents associated with the tables to be
        retrieved.
        - **n** (`int`): The multiplicative factor of `k` to pool more relevant
        documents for the hybrid retrieval process.
        - **alpha** (`float`): The weighting factor of the vector and full-text
        retrievers within a hybrid index. Lower `alpha` gives more weight to
        the vector retriever.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        logger.info(f"Querying the index {index_name}")

        if isinstance(queries, str):
            queries = [queries]

        try:
            vector_retriever, fulltext_retriever = self.__get_retrievers(index_name)
        except ValueError:
            error_message = f"Index with name {index_name} does not exist."
            logger.debug(error_message)
            return Response(
                status=ResponseStatus.ERROR,
                message=error_message,
            ).to_json()

        # Retrieve more documents (k * n) to pool more relevant documents
        increased_k = min(k * n, len(fulltext_retriever.corpus))
        queries_tokens: list[Tokenized] = [bm25s.tokenize(query) for query in queries]

        bm25s.tokenize(queries, stemmer=self.stemmer, show_progress=False)

        logger.info("=> Encoding the queries")
        if isinstance(self.embedding_model, OpenAI):
            queries_embeddings = prompt_openai_embed(
                self.embedding_model,
                queries,
            )
        else:
            queries_embeddings: list[list[float]] = self.embedding_model.encode(
                queries, show_progress_bar=False
            ).tolist()

        queries_tables: list[dict[str, str | list[str]]] = []
        for query_idx, query in enumerate(tqdm(queries, desc="Processing the queries")):
            results, scores = fulltext_retriever.retrieve(
                queries[query_idx], k=increased_k, show_progress=False
            )
            bm25_res = (results, scores)
            vec_res = vector_retriever.query(
                query_embeddings=[queries_embeddings[query_idx]], n_results=increased_k
            )

            all_nodes = self.__hybrid_retriever(
                fulltext_retriever,
                vector_retriever,
                bm25_res,
                vec_res,
                increased_k,
                query,
                alpha,
                queries_tokens[query_idx],
                queries_embeddings[query_idx],
            )

            all_nodes = all_nodes[:k]
            tables = []
            for table, score, content in all_nodes:
                table = table.split("_SEP_")[0]
                tables.append(table)
            queries_tables.append(
                {
                    "query": query,
                    "retrieved_tables": tables,
                }
            )

        logger.info("All queries have been processed.")
        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"Queries successful for index {index_name}",
            data=queries_tables,
        ).to_json()

    def __get_retrievers(self, index_name: str) -> tuple[Collection, bm25s.BM25]:
        """
        Get both vector and full-text retrievers of the index `index_name`.

        ## Args
        - **index_name** (`str`): The name of the hybrid index.

        ## Returns
        - `tuple[Collection, bm25s.BM25]`: The vector and full-text retrievers.
        """
        chroma_client = PersistentClient(self.vector_index_path)
        vector_retriever = chroma_client.get_collection(index_name)
        fulltext_retriever = bm25s.BM25.load(
            os.path.join(self.fulltext_index_path, index_name),
            load_corpus=True,
        )
        return (vector_retriever, fulltext_retriever)

    def __hybrid_retriever(
        self,
        bm25_retriever: bm25s.BM25,
        vec_retriever: Collection,
        bm25_res: tuple[ndarray, ndarray],
        vec_res: QueryResult,
        k: int,
        query: str,
        alpha: float,
        query_tokens: Tokenized,
        query_embedding: list[float],
    ) -> list[tuple[str, float, str]]:
        """
        Generates a hybrid index with name `index_name` for a given `table_ids`.

        ## Args
        - **bm25_retriever** (`BM25`): The full-text retriever within the hybrid
        index.
        - **vec_retriever** (`Collection`): The vector retriever within the hybrid
        index.
        - **bm25_res** (`tuple[ndarray, ndarray]`): Retrieval results from the
        full-text retriever.
        - **vec_res** (`QueryResult`): Retrieval results from the vector retriever.
        - **k** (`int`): The number of documents retrieved from both retrievers.
        - **query** (`str`): The query.
        - **alpha** (`float`): The weighting factor of the vector and full-text
        retrievers within a hybrid index. Lower `alpha` gives more weight to
        the vector retriever.
        - **query_tokens** (`Tokenized`): The tokenized query.
        - **query_embeddings** (`list[float]`): The embedding of the query

        ## Returns
        - `list[tuple[str, float, str]]`: The result of the hybrid search.
        """
        vec_ids = {vec_id for vec_id in vec_res["ids"][0]}
        bm25_ids = {node["metadata"]["table"] for node in bm25_res[0][0]}
        dictionary_id_bm25: dict[str, int] = {
            datum["metadata"]["table"]: idx
            for idx, datum in enumerate(bm25_retriever.corpus)
        }

        processed_nodes_bm25 = self.__process_nodes_bm25(
            bm25_res,
            list(vec_ids - bm25_ids),
            dictionary_id_bm25,
            bm25_retriever,
            query_tokens,
        )
        processed_nodes_vec = self.__process_nodes_vec(
            vec_res, list(bm25_ids - vec_ids), vec_retriever, query_embedding
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

        reranked_nodes = self.__rerank(sorted_nodes, query)
        return reranked_nodes

    def __process_nodes_bm25(
        self,
        items: tuple[ndarray, ndarray],
        missing_ids: list[str],
        dictionary_id_bm25: dict[str, int],
        bm25_retriever: bm25s.BM25,
        query_tokens: Tokenized,
    ):
        """
        Processes the retrieval results of the full-text retriever for the purpose
        of hybrid search (augment the results with missing IDs of documents
        retrieved from the vector index).

        ## Args
        - **items** (`tuple[ndarray, ndarray]`): Retrieval results from the
        full-text retriever.
        - **missing_ids** (`list[str]`): The IDs available in the retrieval results
        of the vector retriever but not full-text retriever.
        - **dictionary_id_bm25** (`dict[str, int]`): The table-document
        associations within the full-text retriever.
        - **bm25_retriever** (`BM25`): The full-text retriever.
        - **query_tokens** (`Tokenized`): The tokenized query.

        ## Returns
        - `dict[str, tuple[float, str]]`: The processed results representing the
        score and document of each document ID.
        """
        results = [node for node in items[0][0]]
        scores = [node for node in items[1][0]]

        extra_results = [
            bm25_retriever.corpus[dictionary_id_bm25[idx]] for idx in missing_ids
        ]
        extra_scores = [
            bm25_retriever.get_scores(
                convert_tokenized_to_string_list(query_tokens)[0]
            )[dictionary_id_bm25[idx]]
            for idx in missing_ids
        ]

        results.extend(extra_results)
        scores.extend(extra_scores)

        max_score = max(scores)
        min_score = min(scores)

        processed_nodes: dict[str, tuple[float, str]] = {
            node["metadata"]["table"]: (
                (
                    1
                    if min_score == max_score
                    else (scores[i] - min_score) / (max_score - min_score)
                ),
                node["text"],
            )
            for i, node in enumerate(results)
        }
        return processed_nodes

    def __process_nodes_vec(
        self,
        items: QueryResult,
        missing_ids: list[str],
        collection: Collection,
        query_embedding: list[float],
    ):
        """
        Processes the retrieval results of the vector retriever for the purpose
        of hybrid search (augment the results with missing IDs of documents
        retrieved from the full-text index).

        ## Args
        - **items** (`QueryResult`): Retrieval results from the vector retriever.
        - **missing_ids** (`list[str]`): The IDs available in the retrieval results
        of the full-text retriever but not vector retriever.
        - **collection** (`dict[str, int]`): The vector retriever
        - **query_embedding** (`list[float]`): The embedding of the query.

        ## Returns
        - `dict[str, tuple[float, str]]`: The processed results representing the
        score and document of each document ID.
        """
        extra_information = collection.get_fast(
            ids=missing_ids, limit=len(missing_ids), include=["documents", "embeddings"]
        )
        items["ids"][0].extend(extra_information["ids"])
        items["documents"][0].extend(extra_information["documents"])
        items["distances"][0].extend(
            cosine(query_embedding, extra_information["embeddings"][i])
            for i in range(len(missing_ids))
        )

        scores: list[float] = [1 - dist for dist in items["distances"][0]]
        documents: list[str] = items["documents"][0]
        ids: list[str] = items["ids"][0]

        max_score = max(scores)
        min_score = min(scores)

        processed_nodes: dict[str, tuple[float, str]] = {
            ids[idx]: (
                (
                    1
                    if min_score == max_score
                    else (scores[idx] - min_score) / (max_score - min_score)
                ),
                documents[idx],
            )
            for idx in range(len(scores))
        }
        return processed_nodes

    def __rerank(
        self,
        nodes: list[tuple[str, float, str]],
        query: str,
    ) -> list[tuple[str, float, str]]:
        """
        Perform re-ranking of documents against the query. Basically, the `LLM
        Judge` classifies whether a document is relevant or not against the query.

        ## Args
        - **nodes** (`list[tuple[str, float, str]]`): The list of tuples, each of
        which consists of ID, relevance score, and document, resulted from the
        hybrid search mechanism.
        - **query** (`str`): The query.

        ## Returns
        - `dict[str, tuple[float, str]]`: The re-ranked results.
        """
        node_tables = [node[0] for node in nodes]

        relevance_prompts = [
            [
                {
                    "role": "user",
                    "content": self.__get_relevance_prompt(
                        node[2],
                        (
                            "content"
                            if node[0].split("_SEP_")[1].startswith("contents")
                            else "context"
                        ),
                        query,
                    ),
                }
            ]
            for node in nodes
        ]

        if isinstance(self.pipe, OpenAI):
            arguments = prompt_openai_llm(
                self.pipe,
                relevance_prompts,
                max_new_tokens=2,
            )
        else:
            arguments = prompt_pipeline(
                self.pipe,
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

    def __get_relevance_prompt(self, desc: str, desc_type: str, query: str):
        """
        Returns relevance prompts for re-ranking purposes. The prompt format is
        slightly different between content summaries and context (metadata).

        ## Args
        - **desc** (`str`): The description of a table, which is either a content
        summary or context (metadata).
        - **desc_type** (`str`): The description type: content or context.
        - **query** (`str`): The query to be compared against.

        ## Returns
        - `str`: A relevance prompt.
        """
        if desc_type == "content":
            return f"""Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{query}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
        elif desc_type == "context":
            return f"""Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{query}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""


if __name__ == "__main__":
    fire.Fire(QueryProcessor)
