import logging

import torch
from openai import OpenAI, OpenAIError
from torch.cuda import OutOfMemoryError
from transformers import set_seed
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("prompting_interface.py")


def truncate_conversation_if_necessary(
    tokenizer: PreTrainedTokenizerBase,
    conversation: list[dict[str, str]],
    context_length: int,
    max_new_tokens: int,
):
    """
    Truncate the text of a conversation according to context_length and max_new_tokens

    ### Parameters:
    - tokenizer (PreTrainedTokenizerBase): A PreTrainedTokenizerBase from HuggingFace
    - conversation (list[dict[str, str]]): A conversation
    - context_length (int): The context length of a model
    - max_new_tokens (int): Max new tokens to be generated

    ### Returns:
    - text (str): The truncated text
    """
    base = [{"role": "user", "content": ""}]
    base_len = len(
        tokenizer.apply_chat_template(base, tokenize=True, add_generation_prompt=True)
    )
    max_tokens = context_length - base_len - max_new_tokens
    tokens = tokenizer.tokenize(conversation[0]["content"])[:max_tokens]
    conversation[0]["content"] = tokenizer.convert_tokens_to_string(tokens)
    return conversation


def remove_unset_generation_configs(generation_configs: dict[str, any]):
    """
    Check whether a conversation is within context_length

    ### Parameters:
    - generation_configs (dict[str, any]): A dictionary of generation configs
    """
    if generation_configs["top_k"] == 0:
        del generation_configs["top_k"]
    if generation_configs["top_p"] == 1.0:
        del generation_configs["top_p"]
    if generation_configs["penalty_alpha"] == 0.0:
        del generation_configs["penalty_alpha"]
    if generation_configs["temperature"] == 0.0:
        del generation_configs["temperature"]


def prompt_pipeline(
    pipe: TextGenerationPipeline,
    conversations: list[list[dict[str, str]]],
    batch_size=2,
    context_length=8192,
    max_new_tokens=512,
    do_sample=False,
    top_k=0,
    top_p=1.0,
    penalty_alpha=0.0,
    temperature=0.0,
):
    """
    Prompt the pipeline with a conversation

    ### Parameters:
    - pipe (TextGenerationPipeline): An initialized pipeline.
    - conversations (list[list[dict[str, str]]]): The data type of the model
    - batch_size (int): The batch size to process conversations
    - context_length (int): The LLM's context length
    - max_new_tokens (int): Max tokens to generate
    - do_sample (bool): Perform sampling or not
    - top_k (int): The number of tokens to consider when sampling
    - top_p (float): Minimum cumulative probability of tokens being considered
    - penalty_alpha (float): The amount of focus being put to ensure non-repetitiveness
    - temperature (float): Control how sharp the distribution (smaller means sharper)

    ### Returns:
    - conversations (list[list[dict[str, str]]]): The conversations with model responses appended
    """
    generation_configs = {
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "penalty_alpha": penalty_alpha,
        "temperature": temperature,
        "pad_token_id": pipe.tokenizer.eos_token_id,
    }
    remove_unset_generation_configs(generation_configs)
    try:
        for i in range(len(conversations)):
            conversations[i] = truncate_conversation_if_necessary(
                pipe.tokenizer, conversations[i], context_length, max_new_tokens
            )
        set_seed(42, deterministic=True)
        answers = pipe(
            conversations, truncation=True, batch_size=batch_size, **generation_configs
        )
        results: list[list[dict[str, str]]] = []
        if isinstance(answers[0], dict):
            answers = [answers]
        for answer in answers:
            results.append(answer[0]["generated_text"])
        return results
    except Exception as error:
        logger.warning(error)
        torch.cuda.empty_cache()
        return [[{"role": "user", "content": ""}]]


def prompt_pipeline_robust(
    pipe: TextGenerationPipeline,
    conversations: list[list[dict[str, str]]],
    batch_size=2,
    context_length=8192,
    max_new_tokens=512,
    do_sample=False,
    top_k=0,
    top_p=1.0,
    penalty_alpha=0.0,
    temperature=0.0,
):
    """
    Prompt the pipeline with a conversation in a robust manner (re-try with lower batch size)

    ### Parameters:
    - pipe (TextGenerationPipeline): An initialized pipeline.
    - conversations (list[list[dict[str, str]]]): List of conversations with {role, content}
    - batch_size (int): The batch size to process conversations
    - context_length (int): The LLM's context length
    - max_new_tokens (int): Max tokens to generate
    - do_sample (bool): Perform sampling or not
    - top_k (int): The number of tokens to consider when sampling
    - top_p (float): Minimum cumulative probability of tokens being considered
    - penalty_alpha (float): The amount of focus being put to ensure non-repetitiveness
    - temperature (float): Control how sharp the distribution (smaller means sharper)

    ### Returns:
    - list[list[dict[str, str]]]: The conversations with model responses appended
    """
    generation_configs = {
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "penalty_alpha": penalty_alpha,
        "temperature": temperature,
        "pad_token_id": pipe.tokenizer.eos_token_id,
    }
    remove_unset_generation_configs(generation_configs)
    batch_size_1_counter = 0
    while True:
        try:
            for i in range(len(conversations)):
                conversations[i] = truncate_conversation_if_necessary(
                    pipe.tokenizer, conversations[i], context_length, max_new_tokens
                )
            set_seed(42, deterministic=True)
            answers = pipe(
                conversations,
                truncation=True,
                batch_size=batch_size,
                **generation_configs,
            )
            results: list[list[dict[str, str]]] = []
            if isinstance(answers[0], dict):
                answers = [answers]
            for answer in answers:
                results.append(answer[0]["generated_text"])
            return (results, batch_size)
        except OutOfMemoryError:
            batch_size = max(batch_size - 10, 1)
            if batch_size == 1:
                batch_size_1_counter += 1
            if batch_size_1_counter == 5:
                raise OutOfMemoryError(
                    "Your GPU memory is probably limited. Try reducing max_batch_size when initializing Pneuma"
                )
            torch.cuda.empty_cache()
            logger.warning(f"Reducing batch size to {batch_size}")


def prompt_openai_llm(
    llm: OpenAI,
    conversations: list[list[dict[str, str]]],
    model="gpt-4o-mini",
    max_new_tokens=512,
    temperature=0.7,
    top_p=1.0,
    retry_attempts=5,
):
    """
    Replicates the Hugging Face pipeline for text generation using OpenAI.

    ### Parameters:
    - conversations (list[list[dict[str, str]]]): List of conversations with {role, content}
    - max_new_tokens (int): Max tokens to generate
    - temperature (float): Control how sharp the distribution (smaller means sharper)
    - top_p (float): Minimum cumulative probability of tokens being considered
    - retry_attempts (int): Number of retries in case of failure

    ### Returns:
    - list[list[dict[str, str]]]: The conversations with model responses appended
    """
    for conv in conversations:
        for attempt in range(retry_attempts):
            try:
                response = llm.chat.completions.create(
                    model=model,
                    messages=conv,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=42,
                )
                conv.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                break
            except OpenAIError as e:
                if attempt < retry_attempts - 1:
                    print(f"OpenAI API error: {e}. Retrying ({attempt + 1}/{retry_attempts})...")
                else:
                    print(f"Failed after {retry_attempts} attempts.")
                    raise e
    return conversations


def prompt_openai_embed(
    embed_model: OpenAI,
    documents: list[str],
    model="text-embedding-3-small",
):
    responses = embed_model.embeddings.create(
        input=documents, model=model,
    )
    embeddings = [i.embedding for i in responses.data]
    return embeddings


if __name__ == "__main__":
    import os
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from pipeline_initializer import initialize_pipeline

    pipe = initialize_pipeline(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch.bfloat16,
    )
    conversations = [[{"role": "user", "content": "Tell me about Illinois!"}]]
    output = prompt_pipeline(
        pipe, conversations, 1, 8192, temperature=None, top_p=None, max_new_tokens=20
    )
    print(output)
    assert output[0][-1]["role"] == "assistant"
