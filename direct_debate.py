import json
import time
import random
import math
from zai import ZhipuAiClient
import os
import pickle
import numpy as np
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
# 定义问题
DEBATE_QUESTION = "如何评价无人机集群的飞行轨迹优化和协同作业能力？"

# 评分标准配置
SCORING_CRITERIA = {
    "flight_control": {
        "name": "飞行控制评分",
        "weight": 0.35,
        "metrics": {
            "trajectory_smoothness": {"weight": 0.3, "name": "轨迹平滑度"},
            "altitude_stability": {"weight": 0.25, "name": "高度稳定性"},
            "speed_consistency": {"weight": 0.25, "name": "速度一致性"},
            "energy_efficiency": {"weight": 0.2, "name": "能源效率"}
        }
    },
    "swarm_coordination": {
        "name": "集群协同评分",
        "weight": 0.4,
        "metrics": {
            "formation_stability": {"weight": 0.3, "name": "编队稳定性"},
            "communication_quality": {"weight": 0.25, "name": "通信质量"},
            "coordination_delay": {"weight": 0.25, "name": "协调延迟"},
            "task_completion": {"weight": 0.2, "name": "任务完成度"}
        }
    },
    "safety_assessment": {
        "name": "安全评估评分",
        "weight": 0.25,
        "metrics": {
            "collision_avoidance": {"weight": 0.4, "name": "避碰能力"},
            "emergency_response": {"weight": 0.3, "name": "应急响应"},
            "risk_management": {"weight": 0.3, "name": "风险管控"}
        }
    }
}

# 生成模拟无人机飞行数据
def generate_drone_flight_data():
    """
    生成模拟的无人机集群飞行数据
    """
    drone_data = {
        "mission_id": "SWARM_001",
        "mission_type": "协同巡逻任务",
        "drone_count": 5,
        "flight_duration": "45分钟",
        "drones": []
    }
    
    # 为每架无人机生成飞行数据
    for i in range(5):
        drone_id = f"UAV_{i+1:03d}"
        
        # 生成飞行轨迹点 (模拟GPS坐标)
        trajectory = []
        base_lat, base_lon = 39.9042, 116.4074  # 北京坐标作为基准
        
        for t in range(0, 2700, 60):  # 45分钟，每分钟一个点
            # 添加一些随机变化模拟真实飞行轨迹
            lat_offset = 0.01 * math.sin(t/300) + random.uniform(-0.002, 0.002)
            lon_offset = 0.01 * math.cos(t/300) + random.uniform(-0.002, 0.002)
            altitude = 100 + 20 * math.sin(t/600) + random.uniform(-5, 5)
            
            trajectory.append({
                "timestamp": t,
                "latitude": base_lat + lat_offset + i * 0.005,
                "longitude": base_lon + lon_offset + i * 0.005,
                "altitude": max(50, altitude),
                "speed": random.uniform(8, 15),
                "heading": (t/10 + i * 72) % 360
            })
        
        # 生成性能指标
        drone_info = {
            "drone_id": drone_id,
            "model": f"DJI_M{300 + i*100}",
            "trajectory": trajectory,
            "performance_metrics": {
                "avg_speed": round(random.uniform(10, 14), 2),
                "max_altitude": round(max([p["altitude"] for p in trajectory]), 1),
                "total_distance": round(random.uniform(25, 35), 2),
                "battery_consumption": round(random.uniform(75, 95), 1),
                "communication_quality": round(random.uniform(85, 98), 1),
                "formation_accuracy": round(random.uniform(88, 96), 1)
            },
            "anomalies": random.choice([
                [],
                ["轻微GPS信号干扰 (T+1200s)"],
                ["短暂通信延迟 (T+800s)"],
                ["风速影响轨迹偏移 (T+1500s)"]
            ])
        }
        
        drone_data["drones"].append(drone_info)
    
    # 添加集群协同指标
    drone_data["swarm_metrics"] = {
        "formation_stability": round(random.uniform(90, 97), 1),
        "collision_avoidance_events": random.randint(0, 3),
        "communication_success_rate": round(random.uniform(94, 99), 1),
        "task_completion_rate": round(random.uniform(92, 100), 1),
        "energy_efficiency": round(random.uniform(85, 93), 1),
        "coordination_delay_avg": round(random.uniform(50, 150), 0)
    }
    
    return drone_data


# 自动评分算法
def calculate_flight_scores(flight_data):
    """
    基于飞行数据计算各项评分
    """
    scores = {}
    
    # 1. 飞行控制评分
    flight_control_scores = {}
    
    # 轨迹平滑度评分 (基于速度变化和航向变化)
    avg_speed_variance = sum([
        sum([abs(p["speed"] - drone["performance_metrics"]["avg_speed"]) 
             for p in drone["trajectory"]]) / len(drone["trajectory"])
        for drone in flight_data["drones"]
    ]) / len(flight_data["drones"])
    trajectory_smoothness = max(0, 100 - avg_speed_variance * 5)
    flight_control_scores["trajectory_smoothness"] = min(100, trajectory_smoothness)
    
    # 高度稳定性评分
    altitude_variances = []
    for drone in flight_data["drones"]:
        altitudes = [p["altitude"] for p in drone["trajectory"]]
        avg_alt = sum(altitudes) / len(altitudes)
        variance = sum([(alt - avg_alt) ** 2 for alt in altitudes]) / len(altitudes)
        altitude_variances.append(variance)
    avg_altitude_variance = sum(altitude_variances) / len(altitude_variances)
    altitude_stability = max(0, 100 - avg_altitude_variance / 10)
    flight_control_scores["altitude_stability"] = min(100, altitude_stability)
    
    # 速度一致性评分
    speed_consistency = max(0, 100 - avg_speed_variance * 3)
    flight_control_scores["speed_consistency"] = min(100, speed_consistency)
    
    # 能源效率评分 (基于电池消耗)
    avg_battery_consumption = sum([
        drone["performance_metrics"]["battery_consumption"] 
        for drone in flight_data["drones"]
    ]) / len(flight_data["drones"])
    energy_efficiency = max(0, 200 - avg_battery_consumption * 2)
    flight_control_scores["energy_efficiency"] = min(100, energy_efficiency)
    
    scores["flight_control"] = flight_control_scores
    
    # 2. 集群协同评分
    swarm_coordination_scores = {}
    
    # 编队稳定性评分
    formation_stability = flight_data["swarm_metrics"]["formation_stability"]
    swarm_coordination_scores["formation_stability"] = formation_stability
    
    # 通信质量评分
    communication_quality = flight_data["swarm_metrics"]["communication_success_rate"]
    swarm_coordination_scores["communication_quality"] = communication_quality
    
    # 协调延迟评分 (延迟越低分数越高)
    coordination_delay = flight_data["swarm_metrics"]["coordination_delay_avg"]
    coordination_delay_score = max(0, 100 - coordination_delay / 2)
    swarm_coordination_scores["coordination_delay"] = coordination_delay_score
    
    # 任务完成度评分
    task_completion = flight_data["swarm_metrics"]["task_completion_rate"]
    swarm_coordination_scores["task_completion"] = task_completion
    
    scores["swarm_coordination"] = swarm_coordination_scores
    
    # 3. 安全评估评分
    safety_assessment_scores = {}
    
    # 避碰能力评分 (基于避碰事件数量)
    collision_events = flight_data["swarm_metrics"]["collision_avoidance_events"]
    collision_avoidance = max(0, 100 - collision_events * 15)
    safety_assessment_scores["collision_avoidance"] = collision_avoidance
    
    # 应急响应评分 (基于异常处理)
    total_anomalies = sum([len(drone["anomalies"]) for drone in flight_data["drones"]])
    emergency_response = max(0, 100 - total_anomalies * 10)
    safety_assessment_scores["emergency_response"] = emergency_response
    
    # 风险管控评分 (综合评估)
    risk_factors = collision_events + total_anomalies
    risk_management = max(0, 100 - risk_factors * 8)
    safety_assessment_scores["risk_management"] = risk_management
    
    scores["safety_assessment"] = safety_assessment_scores
    
    return scores

def calculate_weighted_scores(scores):
    """
    计算加权综合评分
    """
    weighted_scores = {}
    total_score = 0
    
    for category, category_data in SCORING_CRITERIA.items():
        category_score = 0
        category_weighted_score = 0
        
        for metric, metric_data in category_data["metrics"].items():
            metric_score = scores[category][metric]
            weighted_metric_score = metric_score * metric_data["weight"]
            category_score += weighted_metric_score
            
        category_weighted_score = category_score * category_data["weight"]
        total_score += category_weighted_score
        
        weighted_scores[category] = {
            "score": category_score,
            "weighted_score": category_weighted_score,
            "details": scores[category]
        }
    
    weighted_scores["total_score"] = total_score
    return weighted_scores

def generate_expert_scoring(agent_name, flight_data, scores):
    """
    为专家生成专业评分和分析
    """
    agent_info = AGENTS[agent_name]
    expert_scoring = {
        "expert": agent_name,
        "expertise": agent_info["expertise"],
        "focused_metrics": {},
        "professional_analysis": "",
        "recommendations": []
    }
    
    # 提取该专家关注的评分指标
    for category in ["flight_control", "swarm_coordination", "safety_assessment"]:
        if category in scores:
            for metric in agent_info["scoring_focus"]:
                if metric in scores[category]:
                    expert_scoring["focused_metrics"][metric] = scores[category][metric]
    
    # 计算专家专业评分 (基于关注指标的加权平均)
    if expert_scoring["focused_metrics"]:
        expert_score = sum(expert_scoring["focused_metrics"].values()) / len(expert_scoring["focused_metrics"])
        expert_scoring["expert_score"] = round(expert_score, 2)
    else:
        expert_scoring["expert_score"] = 0
    
    return expert_scoring

def generate_scoring_report(flight_data, scores, weighted_scores, expert_scorings):
    """
    生成综合评分报告
    """
    report = {
        "evaluation_summary": {
            "total_score": round(weighted_scores["total_score"], 2),
            "grade": get_performance_grade(weighted_scores["total_score"]),
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "drone_count": len(flight_data["drones"]),
            "mission_duration": flight_data["mission_info"]["duration_minutes"]
        },
        "category_scores": {},
        "expert_evaluations": expert_scorings,
        "detailed_metrics": scores,
        "recommendations": generate_recommendations(weighted_scores, expert_scorings),
        "performance_analysis": generate_performance_analysis(weighted_scores)
    }
    
    # 分类评分详情
    for category, data in weighted_scores.items():
        if category != "total_score":
            report["category_scores"][category] = {
                "name": SCORING_CRITERIA[category]["name"],
                "score": round(data["score"], 2),
                "weighted_score": round(data["weighted_score"], 2),
                "weight": SCORING_CRITERIA[category]["weight"],
                "grade": get_performance_grade(data["score"])
            }
    
    return report

def get_performance_grade(score):
    """
    根据分数获取等级
    """
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "中等"
    elif score >= 60:
        return "及格"
    else:
        return "不及格"

def generate_recommendations(weighted_scores, expert_scorings):
    """
    生成改进建议
    """
    recommendations = []
    
    # 基于分数生成建议
    for category, data in weighted_scores.items():
        if category != "total_score" and data["score"] < 80:
            category_name = SCORING_CRITERIA[category]["name"]
            if data["score"] < 60:
                recommendations.append(f"{category_name}表现不佳，需要重点改进")
            elif data["score"] < 80:
                recommendations.append(f"{category_name}有提升空间，建议优化")
    
    # 基于专家评分生成建议
    for expert_scoring in expert_scorings:
        if expert_scoring["expert_score"] < 75:
            recommendations.append(f"建议重点关注{expert_scoring['expert']}提出的专业建议")
    
    return recommendations

def generate_performance_analysis(weighted_scores):
    """
    生成性能分析
    """
    analysis = {
        "strengths": [],
        "weaknesses": [],
        "overall_assessment": ""
    }
    
    # 分析优势和劣势
    for category, data in weighted_scores.items():
        if category != "total_score":
            category_name = SCORING_CRITERIA[category]["name"]
            if data["score"] >= 85:
                analysis["strengths"].append(f"{category_name}表现优秀")
            elif data["score"] < 70:
                analysis["weaknesses"].append(f"{category_name}需要改进")
    
    # 总体评估
    total_score = weighted_scores["total_score"]
    if total_score >= 85:
        analysis["overall_assessment"] = "无人机集群整体表现优秀，各项指标均达到较高水平"
    elif total_score >= 75:
        analysis["overall_assessment"] = "无人机集群表现良好，部分指标仍有优化空间"
    elif total_score >= 65:
        analysis["overall_assessment"] = "无人机集群表现中等，需要在多个方面进行改进"
    else:
        analysis["overall_assessment"] = "无人机集群表现不佳，需要全面优化和改进"
    
    return analysis


def call_glm_api(prompt, api_key, agent_role=""):
    base_url = os.environ.get("ZHIPU_BASE_URL", "").strip()
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = ZhipuAiClient(**client_kwargs)
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

def construct_message(agents_responses, question, agent_id):
    if not agents_responses:
        return f"请回答以下问题: {question}\n请提供详细的分析和你的观点。"
    
    prefix_string = f"问题是: {question}\n\n其他智能体的回答如下:\n\n"
    
    for i, response in enumerate(agents_responses):
        if i != agent_id:  # 不包含自己之前的回答
            prefix_string += f"智能体 {i+1} 的回答:\n{response}\n\n"
    
    prefix_string += f"作为智能体 {agent_id+1}，请考虑其他智能体的观点，提供对你的问题的看法。你可以同意、反驳或补充其他智能体的观点。"
    return prefix_string


def load_trajectory_from_pkl(file_path):
    """
    从 trajectory.pkl 加载轨迹并转换为 flight_data 结构
    支持 numpy.ndarray、dict、list 等常见格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到轨迹文件: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 提取数组（支持嵌套结构）
    preferred_keys = {"trajectory", "positions", "observations", "coords", "path"}
    found = []

    def is_numeric(x):
        return isinstance(x, (int, float, np.integer, np.floating))

    def is_numeric_list(lst):
        return isinstance(lst, list) and len(lst) > 0 and all(is_numeric(e) for e in lst)

    def is_numeric_2d_list(lst):
        return isinstance(lst, (list, tuple)) and len(lst) > 0 and all(
            isinstance(e, (list, tuple)) and len(e) > 0 and all(is_numeric(v) for v in e)
            for e in lst
        )

    def collect(obj, path=""):
        if isinstance(obj, np.ndarray):
            found.append((obj, path))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else str(k)
                collect(v, new_path)
        elif isinstance(obj, (list, tuple)):
            if is_numeric_2d_list(obj) or is_numeric_list(obj):
                found.append((np.array(obj, dtype=np.float64), path))
            else:
                for i, v in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    collect(v, new_path)

    collect(data)

    arr = None
    if found:
        # 优先选择路径中包含常见键名的数组，其次选择最大数组
        prioritized = [a for a in found if any(k in a[1] for k in preferred_keys)]
        if prioritized:
            arr = max(prioritized, key=lambda x: x[0].size)[0]
        else:
            arr = max(found, key=lambda x: x[0].size)[0]

    if arr is None:
        print("⚠️ 未能识别 trajectory.pkl 中的轨迹数组，改用模拟数据")
        return generate_drone_flight_data()

    # 轨迹形状处理：至少需要 X、Y，Z 可选
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    elif arr.shape[1] < 2:
        # 如果列不足，重复或填充到2列
        arr = np.pad(arr, ((0,0),(0, max(0,2-arr.shape[1]))), mode='edge')

    # 生成单无人机的轨迹点
    traj = []
    total_distance = 0.0
    prev_xy = None
    max_speed = 0.0

    # 时间步假设为1秒
    for i in range(arr.shape[0]):
        x = float(arr[i, 0])
        y = float(arr[i, 1])
        alt = float(arr[i, 2]) if arr.shape[1] >= 3 else 120.0 + 5.0 * np.sin(i / 20.0)

        if prev_xy is None:
            speed = 0.0
            heading_deg = 0.0
        else:
            dx = x - prev_xy[0]
            dy = y - prev_xy[1]
            step_dist = float(np.hypot(dx, dy))
            total_distance += step_dist
            speed = step_dist  # 单位时间步距离，视为速度
            heading_rad = np.arctan2(dy, dx)
            heading_deg = (np.degrees(heading_rad) + 360.0) % 360.0
            max_speed = max(max_speed, speed)
        prev_xy = (x, y)

        traj.append({
            "gps": {"lat": x, "lon": y},
            "altitude": alt,
            "speed": speed,
            "heading": heading_deg,
            "timestamp": i
        })

    avg_speed = float(np.mean([p["speed"] for p in traj]))
    battery_consumption = round(total_distance * 0.02 + len(traj) * 0.005, 2)  # 简单估算

    flight_data = {
        "mission_id": f"TRJ-{time.strftime('%Y%m%d%H%M%S')}",
        "mission_type": "轨迹分析",
        "drone_count": 1,
        "flight_duration": f"{len(traj)} 步",
        "mission_info": {"duration_minutes": max(1, len(traj)//2)},
        "drones": [
            {
                "id": "DRONE-001",
                "trajectory": traj,
                "performance_metrics": {
                    "avg_speed": round(avg_speed, 3),
                    "max_speed": round(max_speed, 3),
                    "battery_consumption": battery_consumption,
                    "distance_traveled": round(total_distance, 3)
                },
                "anomalies": []
            }
        ],
        "swarm_metrics": {
            "formation_stability": 100.0,  # 单机近似满分
            "communication_success_rate": 100.0,
            "coordination_delay_avg": 10.0,
            "task_completion_rate": 100.0,
            "collision_avoidance_events": 0
        }
    }

    return flight_data

def main():
    api_key = 'd2811fc4f03f48f2bb547d6a6b3378f4.GtaNMZOyqulNGa1L'
    print("🚁 正在加载轨迹文件: c:\\Users\\bafs\\Desktop\\llm_multiagent_debate-main\\trajectory.pkl")
    flight_data = load_trajectory_from_pkl(r"c:\\Users\\bafs\\Desktop\\llm_multiagent_debate-main\\trajectory.pkl")
    print("✅ 已从 trajectory.pkl 加载轨迹数据，并转换为评估结构")

    
    
    # 保存飞行数据
    with open("drone_flight_data.json", "w", encoding="utf-8") as f:
        json.dump(flight_data, f, ensure_ascii=False, indent=2)
    print("✅ 飞行数据已保存到 drone_flight_data.json")
    
    # 设置辩论参数
    agents = 3  # 3个智能体
    rounds = 2  # 2轮辩论
    
    # 智能体角色设定 - 移动到main函数开始处
    global AGENTS
    AGENTS = {
        "无人机飞行控制专家": {
            "role": "无人机飞行控制专家",
            "expertise": "飞行控制系统、轨迹规划、导航算法",
            "scoring_focus": ["trajectory_smoothness", "altitude_stability", "speed_consistency", "energy_efficiency"],
            "evaluation_prompt": "作为无人机飞行控制专家，请重点评估飞行轨迹的平滑度、高度稳定性、速度一致性和能源效率。"
        },
        "集群协同算法专家": {
            "role": "集群协同算法专家", 
            "expertise": "集群智能、协同控制、通信协议",
            "scoring_focus": ["formation_stability", "communication_quality", "coordination_delay", "task_completion"],
            "evaluation_prompt": "作为集群协同算法专家，请重点评估编队稳定性、通信质量、协调延迟和任务完成度。"
        },
        "航空安全专家": {
            "role": "航空安全专家",
            "expertise": "飞行安全、风险评估、应急处理",
            "scoring_focus": ["collision_avoidance", "emergency_response", "risk_management"],
            "evaluation_prompt": "作为航空安全专家，请重点评估避碰能力、应急响应和风险管控能力。"
        }
    }
    
    # 保持向后兼容的角色描述
    agent_roles = [
        "作为一名无人机飞行控制专家，从飞行轨迹优化角度",
        "作为一名集群协同算法专家，从多机协作角度", 
        "作为一名航空安全专家，从飞行安全和风险评估角度"
    ]
    
    # 初始化智能体回答
    agents_responses = [[] for _ in range(agents)]
    
    print(f"\n🤖 智谱GLM-4.6 无人机集群评估系统")
    print(f"辩论问题: {DEBATE_QUESTION}\n")
    
    # 显示飞行数据摘要
    print("📊 飞行数据摘要:")
    print(f"  - 任务ID: {flight_data['mission_id']}")
    print(f"  - 任务类型: {flight_data['mission_type']}")
    print(f"  - 无人机数量: {flight_data['drone_count']} 架")
    print(f"  - 飞行时长: {flight_data['flight_duration']}")
    print(f"  - 集群稳定性: {flight_data['swarm_metrics']['formation_stability']}%")
    print(f"  - 任务完成率: {flight_data['swarm_metrics']['task_completion_rate']}%")
    
    print("\n参与评估的专家:")
    for i, role in enumerate(agent_roles):
        print(f"  专家 {i+1}: {role}")
    print("\n" + "="*60)
    
    # 生成飞行数据摘要
    flight_summary = f"""
基于以下无人机集群飞行数据进行分析：
- 任务类型：{flight_data['mission_type']}
- 无人机数量：{flight_data['drone_count']} 架
- 飞行时长：{flight_data['flight_duration']}
- 集群稳定性：{flight_data['swarm_metrics']['formation_stability']}%
- 通信成功率：{flight_data['swarm_metrics']['communication_success_rate']}%
- 任务完成率：{flight_data['swarm_metrics']['task_completion_rate']}%
- 避碰事件：{flight_data['swarm_metrics']['collision_avoidance_events']} 次
- 平均协调延迟：{flight_data['swarm_metrics']['coordination_delay_avg']} ms"""
    
    # 更新智能体循环以使用新的AGENTS字典
    agent_names = list(AGENTS.keys())
    
    # 生成自动评分
    print("正在计算飞行数据评分...")
    scores = calculate_flight_scores(flight_data)
    weighted_scores = calculate_weighted_scores(scores)
    
    # 生成专家评分
    expert_scorings = []
    for agent_name in agent_names:
        expert_scoring = generate_expert_scoring(agent_name, flight_data, scores)
        expert_scorings.append(expert_scoring)
    
    print(f"自动评分完成，总分: {weighted_scores['total_score']:.2f}")
    
    # 进行辩论
    for round_idx in range(rounds):
        print(f"\n🔥 第 {round_idx+1} 轮评估")
        print("-" * 40)
        
        for agent_idx in range(agents):
            agent_name = agent_names[agent_idx]
            agent_info = AGENTS[agent_name]
            
            print(f"\n💭 {agent_name} 正在分析...")
            
            # 构建提示，包含飞行数据和评分
            if round_idx == 0:
                # 第一轮：基于飞行数据和自动评分进行分析
                prompt = f"""
{agent_info['evaluation_prompt']}

{flight_summary}

自动评分结果：
- 总分：{weighted_scores['total_score']:.2f}
- 飞行控制：{weighted_scores['flight_control']['score']:.2f}
- 集群协同：{weighted_scores['swarm_coordination']['score']:.2f}  
- 安全评估：{weighted_scores['safety_assessment']['score']:.2f}

您的专业评分：{expert_scorings[agent_idx]['expert_score']:.2f}
关注指标：{expert_scorings[agent_idx]['focused_metrics']}

请回答以下问题: {DEBATE_QUESTION}
请从您的专业角度分析这次无人机集群飞行的表现，并提供详细的评价和改进建议。"""
            else:
                # 后续轮次，考虑其他专家的分析
                last_round_responses = [agents_responses[i][round_idx-1] for i in range(agents)]
                prompt = construct_message(last_round_responses, DEBATE_QUESTION, agent_idx)
                prompt += f"\n\n请结合飞行数据（总分{weighted_scores['total_score']:.2f}，您的专业评分：{expert_scorings[agent_idx]['expert_score']:.2f}）进行分析。"
            
            # 调用API获取回答
            print("正在调用智谱GLM-4.6 API...")
            response = call_glm_api(prompt, api_key, agent_info['role'])

            agents_responses[agent_idx].append(response)
            
            print(f"\n📢 {agent_name} 的评估:")
            print(f"{response}")
            print("-" * 40)
            time.sleep(2)  # 避免API调用过于频繁
    
    # 生成综合评分报告
    if "mission_info" not in flight_data:
        flight_data["mission_info"] = {"duration_minutes": 45}
    scoring_report = generate_scoring_report(flight_data, scores, weighted_scores, expert_scorings)
    
    # 保存评估结果
    evaluation_result = {
        "question": DEBATE_QUESTION,
        "flight_data": flight_data,
        "expert_roles": agent_names,
        "expert_evaluations": agents_responses,
        "scoring_report": scoring_report,
        "model": "glm-4.6",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("drone_swarm_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 无人机集群评估完成！")
    print("📄 评估结果已保存到 drone_swarm_evaluation_result.json")
    print("📄 飞行数据已保存到 drone_flight_data.json")
    
    # 显示评分报告摘要
    print(f"\n📊 综合评分报告:")
    print(f"  - 总体评分: {scoring_report['evaluation_summary']['total_score']} ({scoring_report['evaluation_summary']['grade']})")
    for category, data in scoring_report['category_scores'].items():
        print(f"  - {data['name']}: {data['score']} ({data['grade']})")
    
    # 显示统计信息
    print(f"\n📊 评估统计:")
    print(f"  - 使用模型: GLM-4.6")
    print(f"  - 参与专家: {agents} 位")
    print(f"  - 评估轮次: {rounds} 轮")
    print(f"  - 总API调用次数: {agents * rounds} 次")
    print(f"  - 无人机数量: {flight_data['drone_count']} 架")
    print(f"  - 飞行时长: {flight_data['flight_duration']}")
    print(f"  - 集群表现: 稳定性{flight_data['swarm_metrics']['formation_stability']}%, 完成率{flight_data['swarm_metrics']['task_completion_rate']}%")

if __name__ == "__main__":
    main()