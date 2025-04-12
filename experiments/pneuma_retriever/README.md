# Pneuma-Retriever

To ingest all documents for all experiment scenarios, run the following commands:

## RQ1: End-to-End Evaluation

```bash
nohup python hybrid_ingest.py -d chembl -sctn schema_narrations -rctn sample_rows -ctx true >> hybrid_ingest-chembl-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d adventure -sctn schema_narrations -rctn sample_rows -ctx true >> hybrid_ingest-adventure-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d public -sctn schema_narrations -rctn sample_rows -ctx true >> hybrid_ingest-public-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d chicago -sctn schema_narrations -rctn sample_rows -ctx true >> hybrid_ingest-chicago-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d fetaqa -sctn schema_narrations -rctn sample_rows -ctx true >> hybrid_ingest-fetaqa-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d bird -sctn schema_narrations -rctn sample_rows -ctx false >> hybrid_ingest-fetaqa-schema_narrations-sample_rows-none.out &
```

To run the experiment, do the following steps:

1. Download the necessary datasets, summaries, models, and benchmarks in the `pneuma_summarizer/summaries`, `data_src`, and `models`.

2. Install the requirements with the following script:
```bash
pip install -r requirements.txt
pip install -r ../benchmark_generator/context/requirements.txt
```

3. Index the summaries using `index_vector.py` and `index_keyword.py`.

4. Produce embeddings for the summaries with `produce_question_embeddings.py`.

5. Run the benchmarks on a baseline. For example, you can run hybrid search with `nohup python -u hybrid_search.py >> hybrid_search.out &`. You may need to adjust some parts in the code (e.g., the re-ranker to be used).
