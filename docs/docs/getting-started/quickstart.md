The simplest way to explore `Pneuma` is by running the [quickstart Jupyter notebook](https://github.com/TheDataStation/pneuma/blob/main/quickstart.ipynb). This notebook walks you through `Pneuma`'s full workflow, from **data registration** to **querying**. For those eager to dive in, here’s a snippet showcasing its functionality:

```python
from src.pneuma import Pneuma

# Initialize Pneuma
out_path = "out_demo/storage"
pneuma = Pneuma(
    out_path=out_path,
    llm_path="Qwen/Qwen2.5-7B-Instruct",
    embed_path="BAAI/bge-base-en-v1.5",
)
pneuma.setup()

# Register dataset & summarize it
data_path = "data_src/sample_data/csv"
pneuma.add_tables(path=data_path, creator="demo_user")
pneuma.summarize()

# Add context (metadata) if available
metadata_path = "data_src/sample_data/metadata.csv"
pneuma.add_metadata(metadata_path=metadata_path)

# Generate index
pneuma.generate_index(index_name="demo_index")

# Query the index
response = pneuma.query_index(
    index_name="demo_index",
    query="Which dataset contains climate issues?",
    k=1,
    n=5,
    alpha=0.5,
)
response = json.loads(response)
query = response["data"]["query"]
retrieved_tables = response["data"]["response"]
```
