"""
run.py gives all the scores over a prediction file.

This code is to generate another jsonl w.r.t. each report from the predction file.

Output: a jsonl. Each json: { id:"", reference: "", prediction: "", rougeL: ""}
"""
import json
import pandas as pd
import evaluate
from tqdm import tqdm

def load_pred_json(file_path):
    ids, preds, refs = [], [], []
    with open(file_path) as f:
        for row_index, l in enumerate(f):
            d = json.loads(l)
            ids.append(d['id'])
            preds.append(d["prediction"])
            refs.append(d["reference"])
    return ids, preds, refs


def generate_row_scores(ids, preds, refs, output_file):
    scorer = evaluate.load("rouge")
    # Prepare data for DataFrame
    data = []
    for id, pred, ref in tqdm(zip(ids, preds, refs)):
        score = scorer.compute(predictions=[pred], references=[ref])["rougeL"]
        data.append({
            "id": id,
            "prediction": pred,
            "reference": ref,
            "rougeL": score
        })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def main(pred_file, output_file):
    ids, preds, refs = load_pred_json(pred_file)
    generate_row_scores(ids, preds, refs, output_file)
    print(f"File saved at {output_file}")


if __name__ == '__main__':
    pred_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_MIMIC/test_merged.jsonl"
    output_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_MIMIC/merged/row_scores.csv"

    main(pred_file, output_file)