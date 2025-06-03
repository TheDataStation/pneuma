import json
import os


def main():
    results_path = ["625", "1250", "2500", "5000", "10330"]

    all_scores = {}

    for batch_path in results_path:
        batch_score = {
            "ingestion": [],
            "generate_index": [],
            "query_index": [],
        }
        scores_path = os.listdir(batch_path)
        for score_path in scores_path:
            with open(os.path.join(batch_path, score_path)) as f:
                score = json.load(f)["results"]
                if "ingestion" in score:
                    batch_score["ingestion"].append(score["ingestion"]["time"])
                if "generate_index" in score:
                    batch_score["generate_index"].append(
                        score["generate_index"]["time"]
                    )
                if "query_index" in score:
                    batch_score["query_index"].append(score["query_index"]["time"])

        for key in batch_score:
            batch_score[key].sort()

        batch_score["query_index"] = batch_score["query_index"][:10]
        all_scores[batch_path] = batch_score

    for key in all_scores:
        print(key)
        print("Ingestion")
        print(all_scores[key]["ingestion"][0])
        print("Generate index")
        print(all_scores[key]["generate_index"][0])
        print("Query index")
        for score in all_scores[key]["query_index"]:
            print(score)
        print("====================================")


if __name__ == "__main__":
    main()
