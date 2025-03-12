# Experiments
This directory contains scripts we use to experiment in the paper (SIGMOD 2025). Below is a high-level overview of what each sub-directory represents.

- `models`: Scripts to download all models (LLMs, embedding models, and re-ranker models).
- `other_systems`: Scripts to test the hit rates of baselines, including LlamaIndex's RAG and full-text search. For Solo, please refer to [the repo](https://github.com/TheDataStation/solo) directly.
- `pneuma_retriever`: Scripts to index content summaries & context, and then perform retrieval.
- `pneuma_summarizer`: Scripts to generate all content summaries (or download previously generated ones), which represent table contents.

We are going to update this README to include specific steps to replicate the experiments using the scripts soon.
