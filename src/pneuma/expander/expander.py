import json
import os

# Let's define the LLM we use, let's say using Qwen
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
from pneuma import Pneuma

class Qwen:
  def __init__(self, model_name = "Qwen/Qwen3-8B"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
      device_map="auto"
    )

  def prompt_model(self, prompt: str):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = self.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False
    )
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

    # conduct text completion
    generated_ids = self.model.generate(
      **model_inputs,
      max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content


def parse_json(json_str: str):
  json_str = json_str.strip()
  if json_str.startswith('```'):
    json_str = json_str[3:]
  if json_str.endswith('```'):
    json_str = json_str[:-3]
  if not json_str.endswith('}'):
    json_str += '}'
  return json.loads(json_str)

# Now, let's define our (basic) question-answering pipeline
import os

class Expander:
  def __init__(self):
    self.llm = Qwen()
    self.pneuma = Pneuma()

  def retrieve(self, queries: list, k: int):
    # initial_result: dict[str, pd.DataFrame] = self.pneuma.query_index(query, k)
    response = self.pneuma.query_index(
      index_name="demo_index",
      queries=queries,
      k=1,
      n=5,
      alpha=0.5
    )
    relevant_tables = json.dumps(json.loads(response), indent=4)
    print(relevant_tables)
    relevant_tables = json.loads(relevant_tables)

    table_names = []
    for result in relevant_tables["data"]:
      table_names += result["retrieved_tables"]

    # With regex, we can simply ask LLM to think of patterns that may exist based on the retrieved_tables
    llm_output = self.llm.prompt_model(f"""You are an expert data scientist.
    Your task is to observe the following table names:
    {table_names}

    If you suspect there are some regex patterns among the names, output them so that we can load more tables with the
    same naming patterns. For example, water-testing-2025 would result in the patterns <replace_with_pattern>-testing-2025, water-<replace_with_pattern>-2025, or water-testing-<replace_with_pattern>.
    Only ever replace exactly one section with the regex pattern. Another example is for something like '5cq6-qygt.csv', we get '[a-zA-Z0-9]{4}-qygt.csv' and '5cq6-[a-zA-Z0-9]{4}.csv'

    Output your answer as a JSON object **without** any extra formatting or explanations.
    Use the following format:
    {{
      "patterns": ["<regex patterns from the input table names.>"]
    }}""")
    print(llm_output)
    patterns: list[str] = parse_json(llm_output)["patterns"]

    print(patterns)
    # Read all table names
    # all_table_names = os.listdir("dataset_path")
    # Filter by regex, load the extra tables, and return them

def main():
    print("Hello from Expander's main method!")
    fire.Fire(Expander)

if __name__ == "__main__":
    main()