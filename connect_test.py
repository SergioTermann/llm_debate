from zai import ZhipuAiClient
import os

api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
if not api_key:
    print("错误：未找到环境变量 ZHIPU_API_KEY。请在运行前设置你的API Key，例如在PowerShell中：$env:ZHIPU_API_KEY=\"YOUR_API_KEY\"")
else:
    client = ZhipuAiClient(api_key=api_key)

# 使用 system + user 的消息格式，调用 glm-4.6
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "作为一名营销专家，请为‘智谱AI开放平台’创作一个吸引人的口号"}
    ],
    temperature=0.6
)

# 直接打印回复内容
print(response.choices[0].message.content)