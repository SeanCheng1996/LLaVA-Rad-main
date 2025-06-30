from openai import OpenAI
client = OpenAI(
    api_key="sk-mfsoglkmixpwsdirmpwetxphavpespdrihkapsuueifntnet",
    base_url="https://api.siliconflow.cn/v1"
)
print(client.batches.list())