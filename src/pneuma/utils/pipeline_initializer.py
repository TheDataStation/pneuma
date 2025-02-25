"""
pipeline_initializer.py

This module provides HuggingFace-based LLM pipeline initialization functionality.
"""
from torch import dtype
from transformers import pipeline


def initialize_pipeline(
    model_path: str, torch_dtype: dtype, context_length=8192,
):
    """
    Initializes a HuggingFace text generation pipeline.

    ## Args
    - **model_path** (str): The path of an LLM for the pipeline.
    - **torch_dtype** (dtype): The data type of the LLM.
    - **context_length** (int): The context length of the LLM.

    ## Returns
    - `TextGenerationPipeline`: The pipeline for text generation.
    """
    pipe = pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
    )

    pipe.tokenizer.model_max_length = context_length
    return pipe


if __name__ == "__main__":
    import torch
    from transformers.pipelines.text_generation import TextGenerationPipeline

    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = initialize_pipeline(model_path, torch.bfloat16)

    assert type(pipe) == TextGenerationPipeline
