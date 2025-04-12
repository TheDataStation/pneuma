import json
import os


def main():
    results_path = "evaluation/"
    batch_names = os.listdir(results_path)
    batch_names.sort(key=lambda x: (len(x), x))

    all_scores = {}

    for batch in batch_names:
        batch_score = {
            "query_index": [],
        }
        scores_path = os.listdir(os.path.join(results_path, batch))
        for score_path in scores_path:
            with open(os.path.join(results_path, batch, score_path)) as f:
                score = json.load(f)["evaluation_time"]
                batch_score["query_index"].append(score)

        for key in batch_score:
            batch_score[key].sort()
        batch_score["query_index"] = batch_score["query_index"][:10]
        all_scores[batch] = batch_score

    for key in all_scores:
        print(key)
        print("Query index")
        for score in all_scores[key]["query_index"]:
            print(score, "", sep="")
        print("====================================")

    print(json.dumps(all_scores, indent=4))


if __name__ == "__main__":
    main()
