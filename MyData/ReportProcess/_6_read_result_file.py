import json


def extract_response_file(file_path):
    """
    file_path: raw processed jsonl file path
    return:
    responses: json. Each item corresponds to one report,
    {
        "id":[
                {
                "topic":"",
                "sentence":["",""],
                },
                ...
              ],
        ...
    }
    """
    responses = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            # if i>5:
            #     break
            cur_res = json.loads(line)

            cur_processed = cur_res['response']['body']['choices'][0]['message']['content']
            cur_processed = cur_processed.strip().removeprefix("```json").removesuffix("```").strip()
            if len(cur_processed) < 1:
                continue

            # print(cur_processed)
            cur_processed = json.loads(cur_processed)

            """
            Merge same topic sentences
            """
            merged_data = {}
            for item in cur_processed:
                topic = item["topic"]
                if "sentence" in item:
                    sentences = item["sentence"]
                else:
                    sentences = item["sentences"]
                if topic not in merged_data:
                    merged_data[topic] = []
                merged_data[topic].extend(sentences)
            merged_data = [{"topic": k, "sentences": v} for k, v in merged_data.items()]

            """
            append to result
            """
            responses[cur_res['custom_id']] = merged_data

            # print(responses)
            # break
    return responses


if __name__ == '__main__':
    responses = extract_response_file(
        "/data/sc159/LLaVARad/MyData/ReportProcess/chexpert_organ_labels/prompts/test_prompts_0_5000_response.jsonl")
    print(responses)
