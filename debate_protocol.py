import math
from difflib import SequenceMatcher
import re


def structured_prompt_first_round(agent_info, flight_summary, weighted_scores, expert_scoring, question, evidence_text=None):
    """
    Build a structured prompt requiring CLAIM/EVIDENCE/COUNTER/SUMMARY/CONFIDENCE blocks.
    Only ASCII characters; encourage numeric references from provided scores.
    Optionally include a compact trajectory evidence DSL.
    """
    expert_score = expert_scoring.get('expert_score')
    expert_score_str = f"{expert_score:.2f}" if isinstance(expert_score, (int, float)) else "N/A"
    # Safely format category scores when some categories are disabled/missing
    total_score = weighted_scores.get('total_score', 0.0)
    fc = weighted_scores.get('flight_control', {}).get('score')
    swc = weighted_scores.get('swarm_coordination', {}).get('score')
    sa = weighted_scores.get('safety_assessment', {}).get('score')
    fc_str = f"{fc:.2f}" if isinstance(fc, (int, float)) else "N/A"
    swc_str = f"{swc:.2f}" if isinstance(swc, (int, float)) else "N/A"
    sa_str = f"{sa:.2f}" if isinstance(sa, (int, float)) else "N/A"
    base = f"""
{agent_info['evaluation_prompt']}

{flight_summary}

Auto scores (reference):
- Total: {total_score:.2f}
- Flight Control: {fc_str}
- Swarm Coordination: {swc_str}
- Safety Assessment: {sa_str}

Your expert score: {expert_score_str}
Focused metrics: {expert_scoring['focused_metrics']}

Question: {question}
"""
    if evidence_text:
        base += f"\nTrajectory Summary (LLM-readable):\n{evidence_text}\n"
    base += (
        "\nPlease answer using the following STRICT structured format (ASCII only):\n"
        "[CLAIM] One-sentence core judgement about this mission.\n"
        "[EVIDENCE] 3-5 bullet points; EACH must explain WHY the result occurs,\n"
        "           and cite concrete evidence from DSL or data: SEG[i], EVENT t=..., ATTN t=...,\n"
        "           WAYPTS (x,y), SCORES fields, or swarm_metrics values. Include numbers.\n"
        "[COUNTER] Anticipate a counterargument and provide a minimal-change rebuttal.\n"
        "[SUMMARY] 3 key takeaways + one actionable recommendation.\n"
        "[CONFIDENCE] A number in [0.00, 1.00].\n"
    )
    return base


def construct_structured_followup(last_round_responses, question, agent_id, weighted_scores, expert_scoring, evidence_text=None):
    """
    Build follow-up prompt for later rounds, including other experts' previous responses.
    Enforce structured output blocks.
    Optionally include a compact trajectory evidence DSL.
    """
    target_id = (agent_id + 1) % max(1, len(last_round_responses))
    header = [
        f"Question: {question}",
        "Previous round responses (summaries/truncated):",
    ]
    for i, resp in enumerate(last_round_responses):
        if i != agent_id:
            header.append(f"- Expert {i+1}: {str(resp)[:400]}")
    expert_score = expert_scoring.get('expert_score')
    expert_score_str = f"{expert_score:.2f}" if isinstance(expert_score, (int, float)) else "N/A"
    header.append(
        f"Reference scores: Total={weighted_scores['total_score']:.2f}, Your expert score={expert_score_str}"
    )
    if evidence_text:
        header.append("Trajectory Summary (LLM-readable):\n" + evidence_text)
    header_text = "\n".join(header)
    format_text = (
        "\nPlease REPLY using STRICT structured format (ASCII only):\n"
        "[CLAIM] One-sentence core judgement.\n"
        "[EVIDENCE] 3-5 items; for EACH, explain WHY and cite DSL/data (SEG/EVENT/ATTN/WAYPTS/SCORES).\n"
        f"[COUNTER] Directly challenge Expert {target_id+1}'s key point with boundary conditions.\n"
        "[SUMMARY] 3 conclusions + one action.\n"
        "[CONFIDENCE] 0.00~1.00.\n"
    )
    return header_text + format_text


def parse_structured_response(text: str) -> dict:
    """Parse a response with [CLAIM]/[EVIDENCE]/[COUNTER]/[SUMMARY]/[CONFIDENCE] blocks.
    Returns a dict with keys: claim, evidence, counter, summary, confidence, raw.
    Tolerates missing blocks and falls back to defaults.
    """
    def extract(tag: str) -> str:
        m = re.search(rf"\[{re.escape(tag)}\](.*?)(?=\n\[|$)", text, re.S)
        return m.group(1).strip() if m else ""

    claim = extract("CLAIM")
    evidence = extract("EVIDENCE")
    counter = extract("COUNTER")
    summary = extract("SUMMARY")
    conf_str = extract("CONFIDENCE")
    confidence = 0.5
    if conf_str:
        nums = re.findall(r"[0-9]*\.?[0-9]+", conf_str)
        if nums:
            try:
                confidence = float(nums[0])
            except Exception:
                confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "claim": claim,
        "evidence": evidence,
        "counter": counter,
        "summary": summary,
        "confidence": confidence,
        "raw": text,
    }


def summary_similarity(text_a: str, text_b: str) -> float:
    """Compute string similarity between two summaries using SequenceMatcher."""
    return SequenceMatcher(None, text_a or "", text_b or "").ratio()


def update_agent_weights(prev_weights, structured_responses, alpha=1.0, beta=1.0):
    """
    Update agent weights using confidence and agreement among summaries.
    logits_i = alpha * confidence_i + beta * agree_i
    agree_i = average pairwise similarity of agent i's summary to others.
    Returns a normalized softmax weight list.
    """
    n = len(structured_responses)
    if n == 0:
        return prev_weights

    summaries = [sr.get("summary", "") for sr in structured_responses]
    confidences = [sr.get("confidence", 0.5) for sr in structured_responses]

    agrees = []
    for i in range(n):
        sims = [SequenceMatcher(None, summaries[i], summaries[j]).ratio() for j in range(n) if j != i]
        agree_i = sum(sims) / max(1, len(sims))
        agrees.append(agree_i)

    logits = [alpha * confidences[i] + beta * agrees[i] for i in range(n)]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    denom = sum(exps) or 1.0
    return [e / denom for e in exps]