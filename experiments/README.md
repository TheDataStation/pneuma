# Experiments
This directory contains scripts we use to experiment in the paper. To run the experiments, do the following:

## Prepare environment

We strongly recommend to create a Conda environment and install the requirements within it.

```bash
conda create --name pneuma-experiment python=3.12.2
conda install -c nvidia cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
pip install -r requirements.txt
```

## Download models and datasets

Download the datasets, benchmarks, and models using the following commands:

```bash
python data_src/tables/downloader.py
bash benchmarks/content/downloader.sh
bash benchmarks/context/downloader.sh
python models/downloader.py -hf "[Your HF TOKEN]"
```

## Run the Experiments

### Step 1: Represent the Datasets as Texts

We represent each table in a dataset by describing its schema using LLMs and randomly sampling $r=5$ rows. We also refer to these documents as content summaries, as they summarize what each table represents. [pneuma_summarizer](pneuma_summarizer/README.md) explains to produce the documents or download previously generated ones.

We are going to update this README to include specific steps to replicate the experiments using the scripts soon. Below are some details of what each directory represents.

- `other_systems`: Scripts to test the hit rates of baselines, including LlamaIndex's RAG and full-text search. For Solo, please refer to [the repo](https://github.com/TheDataStation/solo) directly.
- `pneuma_retriever`: Scripts to index content summaries & context, and then perform retrieval.
- `pneuma_summarizer`: Scripts to generate all content summaries (or download previously generated ones), which represent table contents.
