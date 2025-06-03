# Pneuma-Retriever

Ingest documents for all experiment scenarios using the following commands:

```bash
nohup python hybrid_ingest.py -d chembl -sctn schema_narrations -rctn sample_rows -ctx true >> output/hybrid_ingest-chembl-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d adventure -sctn schema_narrations -rctn sample_rows -ctx true >> output/hybrid_ingest-adventure-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d public -sctn schema_narrations -rctn sample_rows -ctx true >> output/hybrid_ingest-public-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d chicago -sctn schema_narrations -rctn sample_rows -ctx true >> output/hybrid_ingest-chicago-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d fetaqa -sctn schema_narrations -rctn sample_rows -ctx true >> output/hybrid_ingest-fetaqa-schema_narrations-sample_rows-ctx.out &
nohup python hybrid_ingest.py -d bird -sctn schema_narrations -rctn sample_rows -ctx false >> output/hybrid_ingest-fetaqa-schema_narrations-sample_rows-none.out &
```

```bash
# ChEMBL
nohup python hybrid_ingest.py -d chembl -sctn schema_narrations -rctn none -ctx false >> output/hybrid_ingest-chembl-schema_narrations-none-false.out &
nohup python hybrid_ingest.py -d chembl -sctn schema_concat -rctn none -ctx false >> output/hybrid_ingest-chembl-schema_concat-none-false.out &
nohup python hybrid_ingest.py -d chembl -sctn none -rctn sample_rows -ctx false >> output/hybrid_ingest-chembl-none-sample_rows-false.out &
nohup python hybrid_ingest.py -d chembl -sctn none -rctn dbreader -ctx false >> output/hybrid_ingest-chembl-none-dbreader-false.out &

# Adventure Works
nohup python hybrid_ingest.py -d adventure -sctn schema_narrations -rctn none -ctx false >> output/hybrid_ingest-adventure-schema_narrations-none-false.out &
nohup python hybrid_ingest.py -d adventure -sctn schema_concat -rctn none -ctx false >> output/hybrid_ingest-adventure-schema_concat-none-false.out &
nohup python hybrid_ingest.py -d adventure -sctn none -rctn sample_rows -ctx false >> output/hybrid_ingest-adventure-none-sample_rows-false.out &
nohup python hybrid_ingest.py -d adventure -sctn none -rctn dbreader -ctx false >> output/hybrid_ingest-adventure-none-dbreader-false.out &

# Public BI
nohup python hybrid_ingest.py -d public -sctn schema_narrations -rctn none -ctx false >> output/hybrid_ingest-public-schema_narrations-none-false.out &
nohup python hybrid_ingest.py -d public -sctn schema_concat -rctn none -ctx false >> output/hybrid_ingest-public-schema_concat-none-false.out &
nohup python hybrid_ingest.py -d public -sctn none -rctn sample_rows -ctx false >> output/hybrid_ingest-public-none-sample_rows-false.out &
nohup python hybrid_ingest.py -d public -sctn none -rctn dbreader -ctx false >> output/hybrid_ingest-public-none-dbreader-false.out &

# Chicago Open
nohup python hybrid_ingest.py -d chicago -sctn schema_narrations -rctn none -ctx false >> output/hybrid_ingest-chicago-schema_narrations-none-false.out &
nohup python hybrid_ingest.py -d chicago -sctn schema_concat -rctn none -ctx false >> output/hybrid_ingest-chicago-schema_concat-none-false.out &
nohup python hybrid_ingest.py -d chicago -sctn none -rctn sample_rows -ctx false >> output/hybrid_ingest-chicago-none-sample_rows-false.out &
nohup python hybrid_ingest.py -d chicago -sctn none -rctn dbreader -ctx false >> output/hybrid_ingest-chicago-none-dbreader-false.out &

# FeTaQA
nohup python hybrid_ingest.py -d fetaqa -sctn schema_narrations -rctn none -ctx false >> output/hybrid_ingest-fetaqa-schema_narrations-none-false.out &
nohup python hybrid_ingest.py -d fetaqa -sctn schema_concat -rctn none -ctx false >> output/hybrid_ingest-fetaqa-schema_concat-none-false.out &
nohup python hybrid_ingest.py -d fetaqa -sctn none -rctn sample_rows -ctx false >> output/hybrid_ingest-chicago-none-sample_rows-false.out &
nohup python hybrid_ingest.py -d fetaqa -sctn none -rctn dbreader -ctx false >> output/hybrid_ingest-fetaqa-none-dbreader-false.out &
```

## RQ-4

```bash
# Temp-1.5 (Qwen Non-Instruct)
nohup python hybrid_ingest.py -d chembl -sctn temperature-1.5-none -rctn none -ctx false >> output/hybrid_ingest-chembl-temperature-1.5-none-none-false.out &
nohup python hybrid_ingest.py -d adventure -sctn temperature-1.5-none -rctn none -ctx false >> output/hybrid_ingest-adventure-temperature-1.5-none-none-false.out &
nohup python hybrid_ingest.py -d public -sctn temperature-1.5-none -rctn none -ctx false >> output/hybrid_ingest-public-temperature-1.5-none-none-false.out &
nohup python hybrid_ingest.py -d chicago -sctn temperature-1.5-none -rctn none -ctx false >> output/hybrid_ingest-chicago-temperature-1.5-none-none-false.out &
nohup python hybrid_ingest.py -d fetaqa -sctn temperature-1.5-none -rctn none -ctx false >> output/hybrid_ingest-fetaqa-temperature-1.5-none-none-false.out &

# Temp-1.5 (Qwen Instruct)
nohup python hybrid_ingest.py -d chembl -sctn temperature-1.5-instruct -rctn none -ctx false >> output/hybrid_ingest-chembl-temperature-1.5-instruct-none-false.out &
nohup python hybrid_ingest.py -d adventure -sctn temperature-1.5-instruct -rctn none -ctx false >> output/hybrid_ingest-adventure-temperature-1.5-instruct-none-false.out &
nohup python hybrid_ingest.py -d public -sctn temperature-1.5-instruct -rctn none -ctx false >> output/hybrid_ingest-public-temperature-1.5-instruct-none-false.out &
nohup python hybrid_ingest.py -d chicago -sctn temperature-1.5-instruct -rctn none -ctx false >> output/hybrid_ingest-chicago-temperature-1.5-instruct-none-false.out &
nohup python hybrid_ingest.py -d fetaqa -sctn temperature-1.5-instruct -rctn none -ctx false >> output/hybrid_ingest-fetaqa-temperature-1.5-instruct-none-false.out &
```

Now to get the hit rates:

```bash

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
