import argparse
from huggingface_hub import hf_hub_download
from huggingface_hub import login


def main(hf_token: str):
    login(hf_token)

    # Download meta-llama/Meta-Llama-3-8B-Instruct
    files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./llama")

    # Download BAAI/bge-base-en-v1.5
    files = [
        "1_Pooling/config.json",
        "config_sentence_transformers.json",
        "config.json",
        "model.safetensors",
        "modules.json",
        "README.md",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ]
    repo_id = "BAAI/bge-base-en-v1.5"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./bge-base")

    # Download BAAI/bge-reranker-v2-m3
    files = [
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    repo_id = "BAAI/bge-reranker-v2-m3"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./bge-reranker")

    # Download mistralai/Mistral-7B-Instruct-v0.3
    files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./mistral")

    # Download Qwen/Qwen2.5-3B-Instruct
    files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    repo_id = "Qwen/Qwen2.5-3B-Instruct"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./qwen-3b")

    # Download Qwen/Qwen2.5-7B-Instruct
    files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./qwen")

    # Download Qwen/Qwen2.5-7B
    files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    repo_id = "Qwen/Qwen2.5-7B"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./qwen-non-instruct")

    # Download HuggingFaceH4/zephyr-7b-beta
    files = [
        "added_tokens.json",
        "all_results.json",
        "config.json",
        "eval_results.json",
        "generation_config.json",
        "model-00001-of-00008.safetensors",
        "model-00002-of-00008.safetensors",
        "model-00003-of-00008.safetensors",
        "model-00004-of-00008.safetensors",
        "model-00005-of-00008.safetensors",
        "model-00006-of-00008.safetensors",
        "model-00007-of-00008.safetensors",
        "model-00008-of-00008.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "train_results.json",
        "trainer_state.json",
    ]
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    for file in files:
        hf_hub_download(repo_id=repo_id, filename=file, local_dir="./zephyr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program downloads models used in the experiments.",
    )
    parser.add_argument("-hf", "--hf-token", required=True, help="Hugging Face token")
    hf = parser.parse_args().hf
    main(hf)
