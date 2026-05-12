import os
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv(Path(".env"))

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

print(response.choices[0].message.content)


# import json
# import os
# import urllib.request
# from pathlib import Path

# import openai
# from dotenv import load_dotenv

# url = "https://api.deepseek.com"

# load_dotenv(Path(".env"))

# headers = {
#     "Content-Type": "application/json",
#     "x-api-key": os.getenv("DEEPSEEK_API_KEY"),
# }

# body = json.dumps(
#     {
#         "model": "deepseek-v4-pro",
#         "messages": [
#             {"role": "user", "content": "What is a neural network in one sentence?"}
#         ],
#         "stream": False,
#         "reasoning_effort": "high",
#         "extra_body": {"thinking": {"type": "enabled"}},
#     }
# ).encode("utf-8")

# req = urllib.request.Request(url, body, headers, method="POST")
# with urllib.request.urlopen(req) as resp:
#     res = json.loads(resp.read())
#     print(res.choices[0].message.content)
