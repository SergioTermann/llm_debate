from zhipuai import ZhipuAI
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = ZhipuAI(api_key="d2811fc4f03f48f2bb547d6a6b3378f4.GtaNMZOyqulNGa1L")
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {
            "role": "system",
            "content": "你是一个有用的AI助手。"
        },
        {
            "role": "user",
            "content": "你好，请介绍一下自己。"
        }
    ]
)
print(response.choices[0].message.content)

