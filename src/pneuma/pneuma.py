"""
pneuma.py

This module serves as an entry point to use `Pneuma`, an LLM-based data discovery
system for tabular data.
"""
import logging
import os
from typing import Optional

import fire
from huggingface_hub import login
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import TextGenerationPipeline
from torch import bfloat16

from pneuma.index_generator.index_generator import IndexGenerator
from pneuma.query_processor.query_processor import QueryProcessor
from pneuma.registrar.registrar import Registrar
from pneuma.summarizer.summarizer import Summarizer
from pneuma.utils.logging_config import configure_logging
from pneuma.utils.pipeline_initializer import initialize_pipeline

configure_logging()
logger = logging.getLogger("Pneuma")


class Pneuma:
    """
    The entry point of `Pneuma`, combining all modules for ther purpose of LLM-based
    table discovery.

    This class provides end-to-end methods from indexing tables (and their metadata,
    if any) to retrieving tables given users' queries.

    ## Attributes
    - **out_path** (`str`): The output folder of Pneuma.
    - **db_path** (`str`): The database path within the output folder of Pneuma.
    - **index_location** (`str`): The index path within the output folder of Pneuma.
    - **hf_token** (`str`): A HuggingFace User Access Tokens.
    - **openai_api_key** (`str`): An OpenAI API key.
    - **use_local_model** (`bool`): The option to use local or third-party models
    (for now, OpenAI models only as both LLM and embedding model).
    - **llm_path** (`str`): The path or name of a local LLM from HuggingFace.
    - **embed_path** (`str`): The path or name of a local embedding model from
    HuggingFace.
    - **max_llm_batch_size** (`int`): Maximum batch size for the dynamic batch
    size selector to explore.
    - **registrar** (`Registrar`): The dataset regisration module.
    - **summarizer** (`Summarizer`): The dataset summarizer module.
    - **index_generator** (`IndexGenerator`): The index generator module.
    - **query_processor** (`QueryProcessor`): The query processor module.
    - **llm** (`OpenAI | TextGenerationPipeline`): The actual LLM (lazily
    initialized).
    - **embed_model** (`OpenAI | SentenceTransformer`): The actual embedding model
    (lazily initialized).
    """

    def __init__(
        self,
        out_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        use_local_model: bool = True,
        llm_path: str = "Qwen/Qwen2.5-7B-Instruct",
        embed_path: str = "BAAI/bge-base-en-v1.5",
        max_llm_batch_size: int = 50,
    ):
        if out_path is None:
            out_path = os.path.join(os.getcwd(), "pneuma-out")
        os.makedirs(out_path, exist_ok=True)

        self.out_path = os.path.abspath(out_path)
        self.db_path = os.path.join(self.out_path, "storage.db")
        self.index_path = os.path.join(self.out_path, "indexes")

        self.hf_token = hf_token
        self.openai_api_key = openai_api_key
        self.use_local_model = use_local_model
        self.llm_path = llm_path
        self.embed_path = embed_path
        self.max_llm_batch_size = max_llm_batch_size

        self.__hf_login()

        self.registrar: Optional[Registrar] = None  # Handles dataset registration
        self.summarizer: Optional[Summarizer] = None  # Summarizes table contents
        self.index_generator: Optional[IndexGenerator] = (
            None  # Generates document indexes
        )
        self.query_processor: Optional[QueryProcessor] = None  # Handles user queries
        self.llm: Optional[OpenAI | TextGenerationPipeline] = None  # Placeholder for LLM
        self.embed_model: Optional[OpenAI | SentenceTransformer] = None  # Placeholder for embedding model

    def __hf_login(self):
        """Logs into Hugging Face if a token is provided."""
        if self.hf_token:
            try:
                login(self.hf_token)
            except ValueError as e:
                logger.warning(f"HF login failed: {e}")

    def __init_registrar(self):
        """Initializes the Registrar module."""
        self.registrar = Registrar(db_path=self.db_path)

    def __init_summarizer(self):
        """Initializes the Summarizer module."""
        self.__init_llm()
        self.__init_embed_model()
        self.summarizer = Summarizer(
            llm=self.llm,
            embed_model=self.embed_model,
            db_path=self.db_path,
            max_llm_batch_size=self.max_llm_batch_size,
        )

    def __init_index_generator(self):
        """Initializes the IndexGenerator module."""
        self.__init_embed_model()
        self.index_generator = IndexGenerator(
            embed_model=self.embed_model,
            db_path=self.db_path,
            index_path=self.index_path,
        )

    def __init_query_processor(self):
        """Initializes the QueryProcessor module."""
        self.__init_llm()
        self.__init_embed_model()
        self.query_processor = QueryProcessor(
            llm=self.llm,
            embed_model=self.embed_model,
            index_path=self.index_path,
        )

    def __init_llm(self):
        """Initializes the LLM."""
        if self.llm is None:
            if self.use_local_model:
                self.llm = initialize_pipeline(
                    self.llm_path,
                    bfloat16,
                    context_length=32768,
                )
                # Specific setting for batching
                self.llm.tokenizer.pad_token_id = self.llm.model.config.eos_token_id
                self.llm.tokenizer.padding_side = "left"
            else:
                self.llm = OpenAI(api_key=self.openai_api_key)

    def __init_embed_model(self):
        """Initializes the embedding model."""
        if self.embed_model is None:
            if self.use_local_model:
                self.embed_model = SentenceTransformer(self.embed_path)
            else:
                self.embed_model = OpenAI(api_key=self.openai_api_key)

    def setup(self) -> str:
        """Setup Pneuma through its `Registrar` module."""
        if self.registrar is None:
            self.__init_registrar()
        return self.registrar.setup()

    def add_tables(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
        accept_duplicates: bool = False,
    ) -> str:
        """
        Registers tables into the database by utilizing the `Registrar` module.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if self.registrar is None:
            self.__init_registrar()
        return self.registrar.add_tables(
            path,
            creator,
            source,
            s3_region,
            s3_access_key,
            s3_secret_access_key,
            accept_duplicates,
        )

    def add_metadata(
        self,
        metadata_path: str,
        table_id: str = "",
    ) -> str:
        """
        Registers metadata into the database by utilizing the `Registrar` module.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if self.registrar is None:
            self.__init_registrar()
        return self.registrar.add_metadata(metadata_path, table_id)

    def summarize(self, table_id: str = None) -> str:
        """
        Summarizes the contents of all unsummarized tables or a specific table
        if `table_id` is provided using the `Summarizer` module.

        ## Args
        - **table_id** (`str`): The specific table ID to be summarized.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if self.summarizer is None:
            self.__init_summarizer()
        return self.summarizer.summarize(table_id)

    def generate_index(
        self, index_name: str, table_ids: list[str] | tuple[str] = None
    ) -> str:
        """
        Generates a hybrid index with name `index_name` for a given `table_ids`
        by utilizing the `IndexGenerator` module.

        ## Args
        - **index_name** (`str`): The name of the index to be generated.
        - **table_ids** (`list[str] | tuple[str]`): The IDs of tables to be indexed
        (to be exact, their content summaries & context/metadata).

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if self.index_generator is None:
            self.__init_index_generator()
        return self.index_generator.generate_index(index_name, table_ids)

    def query_index(
        self,
        index_name: str,
        queries: list[str] | str,
        k: int = 1,
        n: int = 5,
        alpha: int = 0.5,
    ) -> str:
        """
        Retrieves tables for the given `queries` against the index `index_name`
        by utilizing the `QueryProcessor` module.

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
        if self.query_processor is None:
            self.__init_query_processor()
        return self.query_processor.query(index_name, queries, k, n, alpha)


def main():
    print("Hello from Pneuma's main method!")
    fire.Fire(Pneuma)


if __name__ == "__main__":
    main()
