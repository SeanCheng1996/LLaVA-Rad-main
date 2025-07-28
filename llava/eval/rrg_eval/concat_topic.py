"""
For topic_seg model, the predictions are based on each topic.
This file will concatenate each topic together for the same picture.

The following operation could be running the commands below:
cd llava/eval/rrg_eval
python run.py the_output.jsonl --output_dir /path/to/some_dir/
"""
import os
import json
from collections import defaultdict

def process_jsonl_files(input_files, output_file):
    """
    Process multiple JSONL files and merge entries by 'id' with unique 'topic's.
    Args:
        input_files (list): List of paths to input .jsonl files.
        output_file (str): Path to save the processed output .jsonl file.
    """
    merged_data = defaultdict(lambda: {"topics": set(), "references": [], "predictions": []})

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                image_id = entry["id"]

                # Extract topic from query (after the last colon)
                query = entry["query"]
                if ":" not in query:
                    continue  # Skip malformed query
                topic = query.split(":")[-1].strip()

                # If topic already processed for this id, skip
                if topic in merged_data[image_id]["topics"]:
                    continue

                merged_data[image_id]["topics"].add(topic)
                merged_data[image_id]["references"].append(entry["reference"])
                merged_data[image_id]["predictions"].append(entry["prediction"])

    # Write output as a new .jsonl file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for image_id, content in merged_data.items():
            out_entry = {
                "id": image_id,
                "reference": " ".join(content["references"]),
                "prediction": " ".join(content["predictions"])
            }
            f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")


# Example usage
if __name__ == "__main__":
    # Replace with your actual list of .jsonl files
    input_files = [
        "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray/test_0.jsonl",
        "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray/test_1.jsonl",
        "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray/test_2.jsonl",
        "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray/test_3.jsonl"
    ]
    output_file = "/data/sc159/LLaVARad/results/topic_seg/llavarad_IUXray/test_merged.jsonl"
    process_jsonl_files(input_files, output_file)
