import json
import time
from zai import ZhipuAiClient
import os

# 定义问题
DEBATE_QUESTION = "如何评价人工智能对未来社会的影响？"

def call_glm_api(prompt, api_key, agent_role=""):
    client = ZhipuAiClient(api_key=api_key)
    messages = [
        {"role": "system", "content": f"你是一个有用的AI助手。{agent_role}".strip()},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="glm-4.6",
        messages=messages,
        temperature=0.6
    )
    return response.choices[0].message.content

def construct_message(agents_responses, question, agent_id, round_idx):
    """
    构建给当前智能体的消息
    """
    if round_idx == 0:
        return f"请回答以下问题: {question}\n请提供详细的分析和你的观点。"
    prefix_string = f"问题是: {question}\n\n其他智能体在前面轮次的回答如下:\n\n"
    for prev_round in range(round_idx):
        prefix_string += f"=== 第 {prev_round + 1} 轮 ===\n"
        for i in range(len(agents_responses)):
            if i != agent_id and len(agents_responses[i]) > prev_round:
                prefix_string += f"智能体 {i+1}: {agents_responses[i][prev_round]}\n\n"
    prefix_string += f"作为智能体 {agent_id+1}，请考虑其他智能体的观点，提供你对问题的看法。你可以同意、反驳或补充其他智能体的观点。"
    return prefix_string

def main():
    api_key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not api_key:
        print("错误：未找到环境变量 ZHIPU_API_KEY。请在运行前设置你的API Key，例如在PowerShell中：$env:ZHIPU_API_KEY=\"YOUR_API_KEY\"")
        return
    agents = 3
    rounds = 2
    agent_roles = [
        "作为一名AI技术专家，从技术发展角度",
        "作为一名社会学家，从社会影响角度",
        "作为一名伦理学家，从道德伦理角度"
    ]
    agents_responses = [[] for _ in range(agents)]
    print(f"🤖 智谱GLM-4.6 多智能体辩论系统")
    print(f"辩论问题: {DEBATE_QUESTION}\n")
    print("参与辩论的智能体:")
    for i, role in enumerate(agent_roles):
        print(f"  智能体 {i+1}: {role}")
    print("\n" + "="*60)
    for round_idx in range(rounds):
        print(f"\n🔥 第 {round_idx+1} 轮辩论")
        print("-" * 40)
        for agent_idx in range(agents):
            role_desc = agent_roles[agent_idx].split('，')[0]
            print(f"\n💭 智能体 {agent_idx+1} ({role_desc}) 正在思考...")
            if round_idx == 0:
                prompt = f"请回答以下问题: {DEBATE_QUESTION}\n请从你的专业角度提供详细的分析和观点。"
            else:
                prompt = construct_message(agents_responses, DEBATE_QUESTION, agent_idx, round_idx)
            print("正在调用智谱GLM-4.6 API...")
            response = call_glm_api(prompt, api_key, agent_roles[agent_idx])
            agents_responses[agent_idx].append(response)
            print(f"\n📢 智能体 {agent_idx+1} 的回答:")
            print(f"{response}")
            print("-" * 40)
            time.sleep(2)
    debate_result = {
        "question": DEBATE_QUESTION,
        "agent_roles": agent_roles,
        "agents_responses": agents_responses,
        "rounds": rounds,
        "model": "glm-4.6",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("glm46_debate_result.json", "w", encoding="utf-8") as f:
        json.dump(debate_result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 辩论结束！")
    print("📄 结果已保存到 glm46_debate_result.json")
    print(f"\n📊 辩论统计:")
    print(f"  - 使用模型: GLM-4.6")
    print(f"  - 参与智能体: {agents} 个")
    print(f"  - 辩论轮次: {rounds} 轮")
    print(f"  - 总API调用次数: {agents * rounds} 次")

if __name__ == "__main__":
    main()