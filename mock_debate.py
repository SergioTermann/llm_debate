import json
import time
import random

# 定义问题
DEBATE_QUESTION = "如何评价人工智能对未来社会的影响？"

# 模拟的智能体回答库
MOCK_RESPONSES = {
    "positive": [
        "人工智能将极大地提升社会生产力，帮助解决医疗、教育、环境等重大问题。AI可以协助医生进行精准诊断，为学生提供个性化教育，优化能源使用效率。这些技术进步将创造更多就业机会，推动经济增长。",
        "我认为AI的积极影响是显而易见的。它能够处理人类无法高效完成的复杂任务，如大数据分析、模式识别等。在科研领域，AI已经帮助发现新药物、预测气候变化。未来AI将成为人类最重要的工具和伙伴。",
        "虽然存在挑战，但AI带来的机遇远大于风险。关键是要建立合适的监管框架和伦理准则。通过人机协作，我们可以发挥各自优势，创造更美好的未来。教育系统也需要适应，培养与AI协作的能力。"
    ],
    "cautious": [
        "人工智能确实有巨大潜力，但我们必须谨慎对待其风险。AI可能导致大规模失业，加剧社会不平等。算法偏见、隐私泄露、自主武器等问题都需要认真考虑。我们需要在发展AI的同时，建立完善的安全保障机制。",
        "我对AI的影响持谨慎乐观态度。技术本身是中性的，关键在于如何使用。我们需要确保AI的发展符合人类价值观，避免技术被滥用。同时要投资于教育和再培训，帮助人们适应AI时代的变化。",
        "AI的发展速度很快，但监管和伦理框架的建立相对滞后。这种不平衡可能带来风险。我建议采用渐进式发展策略，在确保安全的前提下推进AI应用。国际合作也很重要，需要全球共同制定AI治理标准。"
    ],
    "balanced": [
        "人工智能对社会的影响是双面的。一方面，它能提高效率、解决复杂问题；另一方面，也带来就业、隐私、伦理等挑战。关键是要平衡发展与安全，确保AI技术造福全人类而不是少数人。",
        "我认为需要从多个维度评估AI的影响。经济层面，AI会重塑产业结构；社会层面，会改变人际交往方式；文化层面，会影响价值观念。我们需要全面的政策框架来引导AI健康发展。",
        "AI的影响取决于我们如何塑造它。如果我们能够建立包容性的AI生态系统，确保技术普惠，那么AI将成为推动社会进步的强大力量。但如果缺乏有效治理，也可能加剧现有问题。选择权在我们手中。"
    ]
}

def get_mock_response(agent_id, round_idx, previous_responses=None):
    """
    根据智能体ID和轮次获取模拟回答
    """
    perspectives = ["positive", "cautious", "balanced"]
    perspective = perspectives[agent_id % 3]
    
    if round_idx == 0:
        # 第一轮，返回基础观点
        return MOCK_RESPONSES[perspective][0]
    else:
        # 后续轮次，基于其他智能体的回答进行回应
        if previous_responses:
            # 简单的回应逻辑
            if agent_id == 0:  # 积极派
                return MOCK_RESPONSES[perspective][round_idx % len(MOCK_RESPONSES[perspective])]
            elif agent_id == 1:  # 谨慎派
                return MOCK_RESPONSES[perspective][round_idx % len(MOCK_RESPONSES[perspective])]
            else:  # 平衡派
                return MOCK_RESPONSES[perspective][round_idx % len(MOCK_RESPONSES[perspective])]
        else:
            return MOCK_RESPONSES[perspective][round_idx % len(MOCK_RESPONSES[perspective])]

def construct_debate_context(agents_responses, question, agent_id, round_idx):
    """
    构建辩论上下文
    """
    if round_idx == 0:
        return f"请回答以下问题: {question}"
    
    context = f"问题: {question}\n\n前面轮次的讨论:\n"
    
    for prev_round in range(round_idx):
        context += f"\n--- 第 {prev_round + 1} 轮 ---\n"
        for i, response in enumerate(agents_responses):
            if len(response) > prev_round:
                context += f"智能体 {i+1}: {response[prev_round]}\n"
    
    context += f"\n现在请作为智能体 {agent_id+1} 继续讨论这个问题。"
    return context

def main():
    # 设置辩论参数
    agents = 3  # 3个智能体
    rounds = 3  # 3轮辩论
    
    # 智能体角色设定
    agent_roles = [
        "AI乐观主义者 - 强调AI的积极影响",
        "AI谨慎派 - 关注AI的潜在风险", 
        "AI平衡派 - 寻求发展与安全的平衡"
    ]
    
    # 初始化智能体回答
    agents_responses = [[] for _ in range(agents)]
    
    print(f"🤖 多智能体辩论系统")
    print(f"辩论问题: {DEBATE_QUESTION}\n")
    
    print("参与辩论的智能体:")
    for i, role in enumerate(agent_roles):
        print(f"  智能体 {i+1}: {role}")
    print("\n" + "="*60)
    
    # 进行辩论
    for round_idx in range(rounds):
        print(f"\n🔥 第 {round_idx+1} 轮辩论")
        print("-" * 40)
        
        for agent_idx in range(agents):
            print(f"\n💭 智能体 {agent_idx+1} ({agent_roles[agent_idx].split(' - ')[0]}) 正在思考...")
            time.sleep(1)  # 模拟思考时间
            
            # 获取模拟回答
            if round_idx == 0:
                response = get_mock_response(agent_idx, round_idx)
            else:
                # 传入之前的回答作为上下文
                previous_round_responses = [agents_responses[i][round_idx-1] for i in range(agents)]
                response = get_mock_response(agent_idx, round_idx, previous_round_responses)
            
            agents_responses[agent_idx].append(response)
            
            print(f"\n📢 智能体 {agent_idx+1} 的观点:")
            print(f"{response}")
            print("-" * 40)
    
    # 生成辩论总结
    print(f"\n🎯 辩论总结")
    print("="*60)
    
    summary = {
        "question": DEBATE_QUESTION,
        "agents": agent_roles,
        "rounds": rounds,
        "detailed_responses": agents_responses,
        "summary": "本次辩论展现了对人工智能影响的多元化观点，从乐观、谨慎到平衡的不同视角，体现了AI发展中需要考虑的复杂因素。"
    }
    
    # 保存辩论结果
    with open("mock_debate_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("✅ 辩论结束！")
    print("📄 详细结果已保存到 mock_debate_result.json")
    
    # 显示简要总结
    print(f"\n📊 本次辩论统计:")
    print(f"  - 参与智能体: {agents} 个")
    print(f"  - 辩论轮次: {rounds} 轮") 
    print(f"  - 总发言次数: {agents * rounds} 次")
    print(f"  - 辩论主题: {DEBATE_QUESTION}")

if __name__ == "__main__":
    main()