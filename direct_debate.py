import json
import time
import random
import math
from zai import ZhipuAiClient
import os
import pickle
import numpy as np
import os
from attention_utils import compute_traj_attention, build_attn_dsl_block

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
# 定义问题
DEBATE_QUESTION = "如何评价无人机集群的飞行轨迹优化和协同作业能力？"

# 评分标准配置
SCORING_CRITERIA = {
    "flight_control": {
        "name": "轨迹评估评分",
        "weight": 0.0,
        "disabled": True,
        "metrics": {
            "heading_smoothness": {"weight": 0.35, "name": "航向平滑度"},
            "path_efficiency": {"weight": 0.30, "name": "路径效率"},
            "sharp_turn_ratio": {"weight": 0.20, "name": "急转比例(越低越好)"},
            "altitude_variation": {"weight": 0.15, "name": "高度波动(越低越好)"}
        }
    },
    "swarm_coordination": {
        "name": "决策与协同评分",
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
        "weight": 0.0,
        "metrics": {
            "collision_avoidance": {"weight": 0.4, "name": "避碰能力"},
            "emergency_response": {"weight": 0.3, "name": "应急响应"},
            "risk_management": {"weight": 0.3, "name": "风险管控"}
        }
    }
}

# 辅助：方向字符串
COMPASS_DIRS = [
    ("N", 0), ("NNE", 22.5), ("NE", 45), ("ENE", 67.5),
    ("E", 90), ("ESE", 112.5), ("SE", 135), ("SSE", 157.5),
    ("S", 180), ("SSW", 202.5), ("SW", 225), ("WSW", 247.5),
    ("W", 270), ("WNW", 292.5), ("NW", 315), ("NNW", 337.5)
]


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

    # 单机任务：协同评分不适用，置零并标记
    if int(flight_data.get("drone_count", 1)) <= 1:
        swarm_coordination_scores["formation_stability"] = 0.0
        swarm_coordination_scores["communication_quality"] = 0.0
        swarm_coordination_scores["coordination_delay"] = 0.0
        swarm_coordination_scores["task_completion"] = 0.0
        swarm_coordination_scores["not_applicable"] = True
    
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
    total_score = 0.0
    
    for category, category_data in SCORING_CRITERIA.items():
        # 跳过已禁用的类别（例如轨迹评估）
        if category_data.get("disabled", False):
            continue
        category_score = 0.0
        category_weighted_score = 0.0
 
         # 协同评分在单机任务下不适用
        not_applicable = bool(scores.get(category, {}).get("not_applicable", False))
         
        if not not_applicable:
            for metric, metric_data in category_data["metrics"].items():
                metric_score = scores[category][metric]
                weighted_metric_score = metric_score * metric_data["weight"]
                category_score += weighted_metric_score
             
            category_weighted_score = category_score * category_data["weight"]
            total_score += category_weighted_score
 
        weighted_scores[category] = {
            "score": category_score,
            "weighted_score": category_weighted_score,
            "details": scores[category],
            "not_applicable": not_applicable
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
        if category == "swarm_coordination" and int(flight_data.get("drone_count", 1)) <= 1:
            continue  # 单机任务：协同评分不适用
        if category in scores:
            for metric in agent_info["scoring_focus"]:
                if metric in scores[category]:
                    expert_scoring["focused_metrics"][metric] = scores[category][metric]
    
    # 计算专家专业评分 (基于关注指标的加权平均)
    if expert_scoring["focused_metrics"]:
        expert_score = sum(expert_scoring["focused_metrics"].values()) / len(expert_scoring["focused_metrics"])
        expert_scoring["expert_score"] = round(expert_score, 2)
    else:
        expert_scoring["expert_score"] = None
        expert_scoring["note"] = "单机任务，协同评分不适用"
    
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
            grade = get_performance_grade(data["score"])
            note = ""
            if data.get("not_applicable"):
                grade = "N/A"
                note = "单机任务，协同评分不适用"
            report["category_scores"][category] = {
                "name": SCORING_CRITERIA[category]["name"],
                "score": round(data["score"], 2),
                "weighted_score": round(data["weighted_score"], 2),
                "weight": SCORING_CRITERIA[category]["weight"],
                "grade": grade,
                "note": note
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
            if data.get("not_applicable"):
                continue
            category_name = SCORING_CRITERIA[category]["name"]
            if data["score"] < 60:
                recommendations.append(f"{category_name}表现不佳，需要重点改进")
            elif data["score"] < 80:
                recommendations.append(f"{category_name}有提升空间，建议优化")
    
    # 基于专家评分生成建议
    for expert_scoring in expert_scorings:
        if expert_scoring.get("expert_score") is not None and expert_scoring["expert_score"] < 75:
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
    """
    生成结构化跟进提示，要求输出固定块以提升信息密度：
    [CLAIM][EVIDENCE][COUNTER][SUMMARY][CONFIDENCE]
    并指定一个质询目标以促使直接反驳。
    """
    if not agents_responses:
        return (
            f"问题: {question}\n\n"
            "请使用以下严格结构化格式(ASCII):\n"
            "[CLAIM] 一句核心判断\n"
            "[EVIDENCE] 3-5条证据(含具体数值)\n"
            "[COUNTER] 针对潜在反驳的最小变更回应\n"
            "[SUMMARY] 3点综合 + 1条行动建议\n"
            "[CONFIDENCE] 0.00~1.00\n"
        )

    target_id = (agent_id + 1) % max(1, len(agents_responses))
    prefix = [
        f"问题: {question}",
        "上一轮其他专家的回答要点(截断):",
    ]
    for i, response in enumerate(agents_responses):
        if i != agent_id:
            preview = str(response).strip().replace("\n", " ")
            prefix.append(f"- 专家{i+1}: {preview[:400]}")
    prefix.append(
        "\n请严格使用以下结构化格式(ASCII):\n"
        "[CLAIM] 一句核心判断\n"
        "[EVIDENCE] 3-5条证据(含具体数值)\n"
        f"[COUNTER] 针对专家{target_id+1}关键论点的反驳或边界条件\n"
        "[SUMMARY] 3点综合 + 1条行动建议\n"
        "[CONFIDENCE] 0.00~1.00\n"
    )
    return "\n".join(prefix)


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

# 提取轨迹点的通用字段
def _extract_xy_alt_speed(traj_point):
    # 支持两种结构：{gps:{lat,lon}} 或 {latitude, longitude}
    if "gps" in traj_point:
        x = float(traj_point["gps"].get("lat", 0.0))
        y = float(traj_point["gps"].get("lon", 0.0))
    else:
        x = float(traj_point.get("latitude", traj_point.get("lat", 0.0)))
        y = float(traj_point.get("longitude", traj_point.get("lon", 0.0)))
    alt = float(traj_point.get("altitude", 0.0))
    speed = float(traj_point.get("speed", 0.0))
    heading = float(traj_point.get("heading", 0.0))
    t = int(traj_point.get("timestamp", 0))
    return x, y, alt, speed, heading, t

# 航向转指南针方向
def _compass_from_heading(h):
    h = (h % 360.0)
    best = "N"; best_diff = 1e9
    for name, deg in COMPASS_DIRS:
        diff = min(abs(h - deg), 360 - abs(h - deg))
        if diff < best_diff:
            best_diff = diff
            best = name
    return best

# 简易 Ramer–Douglas–Peucker 算法
def _rdp(points, epsilon=0.001):
    # 简易 Ramer–Douglas–Peucker，多数场景够用
    if len(points) < 3:
        return points
    def _perp_dist(pt, a, b):
        ax, ay = a; bx, by = b; px, py = pt
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        cx, cy = ax + t * dx, ay + t * dy
        return math.hypot(px - cx, py - cy)
    def _rdp_rec(pts):
        if len(pts) <= 2:
            return pts
        a, b = pts[0], pts[-1]
        idx, dmax = 0, -1
        for i in range(1, len(pts) - 1):
            d = _perp_dist(pts[i], a, b)
            if d > dmax:
                idx, dmax = i, d
        if dmax > epsilon:
            left = _rdp_rec(pts[:idx+1])
            right = _rdp_rec(pts[idx:])
            return left[:-1] + right
        else:
            return [a, b]
    return _rdp_rec(points)

# 为 LLM 生成轨迹摘要
def summarize_trajectory_for_llm(flight_data, segments=6, max_events=10, rdp_epsilon=0.001, max_waypoints=8):
    traj = flight_data.get("drones", [{}])[0].get("trajectory", [])
    if not traj:
        return {"meta": {}, "stats": {}, "segments": [], "events": [], "waypoints": []}
    xs, ys, alts, speeds, headings, times = [], [], [], [], [], []
    for p in traj:
        x, y, alt, spd, hdg, t = _extract_xy_alt_speed(p)
        xs.append(x); ys.append(y); alts.append(alt); speeds.append(spd); headings.append(hdg); times.append(t)
    n = len(traj)
    duration_s = (times[-1] - times[0]) if times else n
    # 距离（近似欧氏）
    dist = 0.0
    turns = 0
    accel_events = 0
    decel_events = 0
    sharp_turns = []
    climb_events = []
    for i in range(1, n):
        dist += math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
        dh = abs(headings[i] - headings[i-1])
        dh = min(dh, 360 - dh)
        if dh >= 30:
            turns += 1
            sharp_turns.append({"t": times[i], "angle_deg": round(dh, 1)})
        dv = speeds[i] - speeds[i-1]
        if dv >= 2.0:
            accel_events += 1
        elif dv <= -2.0:
            decel_events += 1
        # 爬升事件：连续上升速率阈值
        rate = (alts[i] - alts[i-1]) / max(1, (times[i] - times[i-1]))
        if rate >= 3.0:
            climb_events.append({"t": times[i], "rate_mps": round(rate, 2)})
    speed_mean = float(np.mean(speeds)) if speeds else 0.0
    speed_std = float(np.std(speeds)) if speeds else 0.0
    speed_max = float(np.max(speeds)) if speeds else 0.0
    alt_mean = float(np.mean(alts)) if alts else 0.0
    alt_std = float(np.std(alts)) if alts else 0.0
    heading_std = float(np.std(headings)) if headings else 0.0
    smoothness_index = 1.0 / (1.0 + heading_std / 90.0)
    # 分段
    seg_len = max(1, n // segments)
    segs = []
    for s in range(segments):
        i0 = s * seg_len
        i1 = min(n, (s+1) * seg_len)
        if i0 >= n:
            break
        v_mean = float(np.mean(speeds[i0:i1])) if i1 > i0 else 0.0
        v_std = float(np.std(speeds[i0:i1])) if i1 > i0 else 0.0
        a_mean = float(np.mean(alts[i0:i1])) if i1 > i0 else 0.0
        a_std = float(np.std(alts[i0:i1])) if i1 > i0 else 0.0
        h_mean = float(np.mean(headings[i0:i1])) if i1 > i0 else 0.0
        dir_str = _compass_from_heading(h_mean)
        turns_seg = 0
        for j in range(i0+1, i1):
            dh = abs(headings[j] - headings[j-1]); dh = min(dh, 360 - dh)
            if dh >= 30: turns_seg += 1
        turn_intensity = "low" if turns_seg <= 1 else ("medium" if turns_seg <= 3 else "high")
        segs.append({
            "t0": int(times[i0]), "t1": int(times[i1-1]) if i1 > i0 else int(times[i0]),
            "dir": dir_str,
            "v_mean": round(v_mean, 2), "v_std": round(v_std, 2),
            "alt_mean": round(a_mean, 1), "alt_std": round(a_std, 1),
            "turn_intensity": turn_intensity
        })
    # 事件时间轴（裁剪）
    events = []
    for e in sharp_turns[:max_events]:
        events.append({"t": e["t"], "type": "sharp_turn", "angle_deg": e["angle_deg"]})
    for e in climb_events[:max_events - len(events)]:
        events.append({"t": e["t"], "type": "climb", "rate_mps": e["rate_mps"]})
    # Waypoints via RDP
    pts = list(zip(xs, ys))
    wps = _rdp(pts, epsilon=rdp_epsilon)
    if len(wps) > max_waypoints:
        step = max(1, len(wps) // max_waypoints)
        wps = wps[::step][:max_waypoints]
    summary = {
        "meta": {"duration_s": int(duration_s), "n_points": n},
        "stats": {
            "distance_km": round(dist / 1000.0, 3),
            "speed_mean_mps": round(speed_mean, 3), "speed_std_mps": round(speed_std, 3), "speed_max_mps": round(speed_max, 3),
            "alt_mean_m": round(alt_mean, 1), "alt_std_m": round(alt_std, 1),
            "turn_count_gt30deg": int(turns),
            "smoothness_index": round(smoothness_index, 3)
        },
        "segments": segs,
        "events": events,
        "waypoints": [[round(a,6), round(b,6)] for a,b in wps]
    }
    return summary

# 将摘要格式化为紧凑 DSL
def format_llm_dsl(summary, scores=None):
    meta = summary.get("meta", {})
    stats = summary.get("stats", {})
    segs = summary.get("segments", [])
    events = summary.get("events", [])
    wps = summary.get("waypoints", [])
    lines = []
    lines.append(f"META: dur={meta.get('duration_s',0)}s, pts={meta.get('n_points',0)}")
    lines.append(
        "STATS: "
        f"L={stats.get('distance_km',0)}km, "
        f"v={stats.get('speed_mean_mps',0)}±{stats.get('speed_std_mps',0)}m/s, "
        f"vmax={stats.get('speed_max_mps',0)}m/s, "
        f"alt={stats.get('alt_mean_m',0)}±{stats.get('alt_std_m',0)}m, "
        f"turns>30°={stats.get('turn_count_gt30deg',0)}, "
        f"smooth={stats.get('smoothness_index',0)}"
    )
    for i, s in enumerate(segs[:8], 1):
        lines.append(
            f"SEG[{i}]: t={s.get('t0',0)}-{s.get('t1',0)}, dir={s.get('dir','')}, "
            f"v={s.get('v_mean',0)}±{s.get('v_std',0)}, alt={s.get('alt_mean',0)}±{s.get('alt_std',0)}, "
            f"turn={s.get('turn_intensity','low')}"
        )
    for e in events[:10]:
        if e.get('type') == 'sharp_turn':
            lines.append(f"EVENT: t={e['t']}, sharp_turn={e['angle_deg']}deg")
        elif e.get('type') == 'climb':
            lines.append(f"EVENT: t={e['t']}, climb={e['rate_mps']}m/s")
    if wps:
        wp_str = "→".join([f"({a},{b})" for a,b in wps])
        lines.append(f"WAYPTS: {wp_str}")
    if scores:
        try:
            sc = scores.get('swarm_coordination', {})
            if sc.get('not_applicable'):
                lines.append("SCORES: COORD=N/A (单机任务)")
            else:
                lines.append(
                    "SCORES: "
                    f"formation_stab={sc.get('formation_stability',0)}, "
                    f"comm_quality={sc.get('communication_quality',0)}, "
                    f"coord_delay={sc.get('coordination_delay',0)}, "
                    f"task_comp={sc.get('task_completion',0)}"
                )
        except Exception:
            pass
    return "\n".join(lines)


def main():
    from debate_protocol import structured_prompt_first_round, construct_structured_followup, parse_structured_response, update_agent_weights, summary_similarity
    api_key = 'd2811fc4f03f48f2bb547d6a6b3378f4.GtaNMZOyqulNGa1L'
    print("🚁 正在加载轨迹文件: c:\\Users\\bafs\\Desktop\\llm_multiagent_debate-main\\trajectory.pkl")
    flight_data = load_trajectory_from_pkl(r"c:\\Users\\bafs\\Desktop\\llm_multiagent_debate-main\\trajectory.pkl")
    print("✅ 已从 trajectory.pkl 加载轨迹数据，并转换为评估结构")


    # 保存飞行数据
    with open("drone_flight_data.json", "w", encoding="utf-8") as f:
        json.dump(flight_data, f, ensure_ascii=False, indent=2)
    print("✅ 飞行数据已保存到 drone_flight_data.json")
    
    # 设置辩论参数
    agents = 0  # 占位，稍后根据AGENTS重设
    rounds = 2  # 2轮辩论
    
    # 智能体角色设定 - 移动到main函数开始处
    global AGENTS
    AGENTS = {

        "群聚能力专家": {
            "role": "群聚能力专家",
            "expertise": "群体行为建模、队形保持、密度控制与邻域交互",
            "scoring_focus": ["formation_stability", "communication_quality", "coordination_delay", "task_completion"],
            "evaluation_prompt": "作为群聚能力专家，请评估群体聚合、队形保持与协同行为质量，结合编队稳定性、通信质量、协调延迟与任务完成度四项指标，引用具体证据并给出低成本的群聚策略改进建议；单机任务下协同相关项视为N/A。"
        },
         "集群协同与通信工程师": {
             "role": "集群协同与通信工程师",
             "expertise": "队形控制、任务分配、网络通信与链路质量",
             "scoring_focus": ["formation_stability", "communication_quality", "coordination_delay", "task_completion"],
             "evaluation_prompt": "作为集群协同与通信工程师，请重点评估编队稳定性、通信质量、协调延迟和任务完成度；若为单机任务请明确协同项为N/A；引用具体指标并提出轻量级协同策略调整。"
         },
         "任务完成度评估专家": {
             "role": "任务完成度评估专家",
             "expertise": "任务规划与执行、里程碑管理、资源与载荷调度",
             "scoring_focus": ["task_completion", "coordination_delay", "communication_quality"],
             "evaluation_prompt": "作为任务完成度评估专家，请结合任务完成率、协调延迟与通信质量，识别影响任务达成的瓶颈，并提出可执行的优化建议；单机任务下协同相关项视为N/A。"
         }
     }
    # 根据新的专家集合重设数量
    agents = len(AGENTS)
     
     # 保持向后兼容的角色描述
    agent_roles = [
        "作为群聚能力专家，从群体聚合与协同角度",
        "作为集群协同与通信工程师，从协作与网络角度", 
        "作为任务完成度评估专家，从任务执行与达成角度"
    ]
    
    # 初始化智能体回答
    agents_responses = [[] for _ in range(agents)]
    agents_structured = [[] for _ in range(agents)]
    agent_weights = [1.0 / agents] * agents
    weights_history = []
    
    print(f"\n🤖 智谱GLM-4.6 无人机集群评估系统")
    print(f"辩论问题: {DEBATE_QUESTION}\n")
        
        # 显示飞行数据摘要
    print("📊 飞行数据摘要:")
    print(f"  - 任务ID: {flight_data['mission_id']}")
    print(f"  - 任务类型: {flight_data['mission_type']}")
    print(f"  - 无人机数量: {flight_data['drone_count']} 架")
    print(f"  - 飞行时长: {flight_data['flight_duration']}")
    print(f"  - 集群稳定性: {'N/A(单机)' if flight_data['drone_count'] <= 1 else str(flight_data['swarm_metrics']['formation_stability']) + '%'}")
    print(f"  - 任务完成率: {'N/A(单机)' if flight_data['drone_count'] <= 1 else str(flight_data['swarm_metrics']['task_completion_rate']) + '%'}")
    
    print("\n参与评估的专家:")
    for i, role in enumerate(agent_roles):
        print(f"  专家 {i+1}: {role}")
    print("\n" + "="*60)
    
    # 生成飞行数据摘要
    swarm_stab_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['formation_stability']}%"
    comm_rate_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['communication_success_rate']}%"
    task_comp_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['task_completion_rate']}%"
    avoid_events_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['collision_avoidance_events']} 次"
    coord_delay_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['coordination_delay_avg']} ms"
    flight_summary = f"""
    基于以下无人机集群飞行数据进行分析：
    - 任务类型：{flight_data['mission_type']}
    - 无人机数量：{flight_data['drone_count']} 架
    - 飞行时长：{flight_data['flight_duration']}
    - 集群稳定性：{swarm_stab_str}
    - 通信成功率：{comm_rate_str}
    - 任务完成率：{task_comp_str}
    - 避碰事件：{avoid_events_str}
    - 平均协调延迟：{coord_delay_str}"""
        
        # 更新智能体循环以使用新的AGENTS字典
    agent_names = list(AGENTS.keys())
        
        # 生成自动评分
    print("正在计算飞行数据评分...")
    scores = calculate_flight_scores(flight_data)
    weighted_scores = calculate_weighted_scores(scores)
    
    # 为LLM生成紧凑轨迹证据DSL
    traj_summary = summarize_trajectory_for_llm(flight_data)
    evidence_text = format_llm_dsl(traj_summary, scores=scores)
    try:
        attn = compute_traj_attention(flight_data.get("drones", [{}])[0].get("trajectory", []))
        evidence_text = evidence_text + "\n" + build_attn_dsl_block(flight_data.get("drones", [{}])[0].get("trajectory", []), attn)
        print("🧪 已生成轨迹摘要DSL(已注入到提示): META/SEG/EVENT/WAYPTS/SCORES/ATTN")
    except Exception:
        print("⚠️ 注意力摘要构建失败，继续使用原始DSL")
    # 打印抽象结果：JSON与DSL
    print("\n====== 轨迹摘要(JSON) ======")
    print(json.dumps(traj_summary, ensure_ascii=False, indent=2))
    print("\n====== 轨迹摘要(DSL) ======")
    print(evidence_text)
    print("="*60)
    
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
                prompt = structured_prompt_first_round(
                    agent_info,
                    flight_summary,
                    weighted_scores,
                    expert_scorings[agent_idx],
                    DEBATE_QUESTION,
                    evidence_text=evidence_text,
                )
            else:
                # 后续轮次，考虑其他专家的分析
                last_round_responses = [agents_responses[i][round_idx-1] for i in range(agents)]
                prompt = construct_structured_followup(
                    last_round_responses,
                    DEBATE_QUESTION,
                    agent_idx,
                    weighted_scores,
                    expert_scorings[agent_idx],
                    evidence_text=evidence_text,
                )
            
            # 调用API获取回答
            print("正在调用智谱GLM-4.6 API...")
            response = call_glm_api(prompt, api_key, agent_info['role'])

            agents_responses[agent_idx].append(response)
            # 解析结构化输出并保存
            try:
                structured = parse_structured_response(response)
            except Exception:
                structured = {"summary": "", "confidence": 0.5, "raw": response}
            agents_structured[agent_idx].append(structured)
            
            print(f"\n📢 {agent_name} 的评估:")
            print(f"{response}")
            print("-" * 40)
            # 额外打印解析后的结构化要点与证据，便于核对原因与引用
            try:
                print("🔎 结构化要点:")
                print(f"CLAIM: {structured.get('claim','')}")
                ev = structured.get('evidence','')
                if ev:
                    print("EVIDENCE:")
                    print(ev)
            except Exception:
                pass
            time.sleep(2)  # 避免API调用过于频繁
    
        # 每轮结束后：更新权重与早停判据（位于轮次循环内）
        try:
            current_structs = [agents_structured[i][round_idx] for i in range(agents)]
        except Exception:
            current_structs = []
        if current_structs:
            agent_weights = update_agent_weights(agent_weights, current_structs, alpha=1.0, beta=1.0)
            weights_history.append(agent_weights)
        if round_idx > 0 and current_structs:
            prev_text = " ".join([agents_structured[i][round_idx-1].get('summary','') for i in range(agents)])
            curr_text = " ".join([s.get('summary','') for s in current_structs])
            sim = summary_similarity(prev_text, curr_text)
            if sim > 0.92:
                print(f"⏹️ 早停: 第{round_idx+1}轮与上一轮相似度 {sim:.2f} 超过阈值 0.92")
                break

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
        "structured_evaluations": agents_structured,
        "debate_weights_history": weights_history,
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
    agents_responses = [[] for _ in range(agents)]
    agents_structured = [[] for _ in range(agents)]
    agent_weights = [1.0 / agents] * agents
    weights_history = []
    
    print(f"\n🤖 智谱GLM-4.6 无人机集群评估系统")
    print(f"辩论问题: {DEBATE_QUESTION}\n")
    
    # 显示飞行数据摘要
    print("📊 飞行数据摘要:")
    print(f"  - 任务ID: {flight_data['mission_id']}")
    print(f"  - 任务类型: {flight_data['mission_type']}")
    print(f"  - 无人机数量: {flight_data['drone_count']} 架")
    print(f"  - 飞行时长: {flight_data['flight_duration']}")
    print(f"  - 集群稳定性: {'N/A(单机)' if flight_data['drone_count'] <= 1 else str(flight_data['swarm_metrics']['formation_stability']) + '%'}")
    print(f"  - 任务完成率: {'N/A(单机)' if flight_data['drone_count'] <= 1 else str(flight_data['swarm_metrics']['task_completion_rate']) + '%'}")
    
    print("\n参与评估的专家:")
    for i, role in enumerate(agent_roles):
        print(f"  专家 {i+1}: {role}")
    print("\n" + "="*60)
    
    # 生成飞行数据摘要
    swarm_stab_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['formation_stability']}%"
    comm_rate_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['communication_success_rate']}%"
    task_comp_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['task_completion_rate']}%"
    avoid_events_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['collision_avoidance_events']} 次"
    coord_delay_str = "N/A(单机)" if flight_data['drone_count'] <= 1 else f"{flight_data['swarm_metrics']['coordination_delay_avg']} ms"
    flight_summary = f"""
    基于以下无人机集群飞行数据进行分析：
    - 任务类型：{flight_data['mission_type']}
    - 无人机数量：{flight_data['drone_count']} 架
    - 飞行时长：{flight_data['flight_duration']}
    - 集群稳定性：{swarm_stab_str}
    - 通信成功率：{comm_rate_str}
    - 任务完成率：{task_comp_str}
    - 避碰事件：{avoid_events_str}
    - 平均协调延迟：{coord_delay_str}"""
    
    # 更新智能体循环以使用新的AGENTS字典
    agent_names = list(AGENTS.keys())
    
    # 生成自动评分
    print("正在计算飞行数据评分...")
    scores = calculate_flight_scores(flight_data)
    weighted_scores = calculate_weighted_scores(scores)
    
    # 为LLM生成紧凑轨迹证据DSL
    traj_summary = summarize_trajectory_for_llm(flight_data)
    evidence_text = format_llm_dsl(traj_summary, scores=scores)
    try:
        attn = compute_traj_attention(flight_data.get("drones", [{}])[0].get("trajectory", []))
        evidence_text = evidence_text + "\n" + build_attn_dsl_block(flight_data.get("drones", [{}])[0].get("trajectory", []), attn)
        print("🧪 已生成轨迹摘要DSL(已注入到提示): META/SEG/EVENT/WAYPTS/SCORES/ATTN")
    except Exception:
        print("⚠️ 注意力摘要构建失败，继续使用原始DSL")
    # 打印抽象结果：JSON与DSL
    print("\n====== 轨迹摘要(JSON) ======")
    print(json.dumps(traj_summary, ensure_ascii=False, indent=2))
    print("\n====== 轨迹摘要(DSL) ======")
    print(evidence_text)
    print("="*60)
    
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
                prompt = structured_prompt_first_round(
                    agent_info,
                    flight_summary,
                    weighted_scores,
                    expert_scorings[agent_idx],
                    DEBATE_QUESTION,
                    evidence_text=evidence_text,
                )
            else:
                # 后续轮次，考虑其他专家的分析
                last_round_responses = [agents_responses[i][round_idx-1] for i in range(agents)]
                prompt = construct_structured_followup(
                    last_round_responses,
                    DEBATE_QUESTION,
                    agent_idx,
                    weighted_scores,
                    expert_scorings[agent_idx],
                    evidence_text=evidence_text,
                )
            
            # 调用API获取回答
            print("正在调用智谱GLM-4.6 API...")
            response = call_glm_api(prompt, api_key, agent_info['role'])

            agents_responses[agent_idx].append(response)
            # 解析结构化输出并保存
            try:
                structured = parse_structured_response(response)
            except Exception:
                structured = {"summary": "", "confidence": 0.5, "raw": response}
            agents_structured[agent_idx].append(structured)
            
            print(f"\n📢 {agent_name} 的评估:")
            print(f"{response}")
            print("-" * 40)
            # 额外打印解析后的结构化要点与证据，便于核对原因与引用
            try:
                print("🔎 结构化要点:")
                print(f"CLAIM: {structured.get('claim','')}")
                ev = structured.get('evidence','')
                if ev:
                    print("EVIDENCE:")
                    print(ev)
            except Exception:
                pass
            time.sleep(2)  # 避免API调用过于频繁
    
        # 每轮结束后：更新权重与早停判据（位于轮次循环内）
        try:
            current_structs = [agents_structured[i][round_idx] for i in range(agents)]
        except Exception:
            current_structs = []
        if current_structs:
            agent_weights = update_agent_weights(agent_weights, current_structs, alpha=1.0, beta=1.0)
            weights_history.append(agent_weights)
        if round_idx > 0 and current_structs:
            prev_text = " ".join([agents_structured[i][round_idx-1].get('summary','') for i in range(agents)])
            curr_text = " ".join([s.get('summary','') for s in current_structs])
            sim = summary_similarity(prev_text, curr_text)
            if sim > 0.92:
                print(f"⏹️ 早停: 第{round_idx+1}轮与上一轮相似度 {sim:.2f} 超过阈值 0.92")
                break

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
        "structured_evaluations": agents_structured,
        "debate_weights_history": weights_history,
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