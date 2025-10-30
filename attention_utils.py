import numpy as np


def compute_traj_attention(traj, alpha=0.6, beta=0.3, gamma=0.1, top_k=8):
    """
    Compute heuristic attention over a trajectory.
    - traj: list of dicts with keys like 'speed', 'heading', 'altitude', 'timestamp'.
    Returns dict with 'weights' (softmax), 'top_indices', and raw 'importance'.
    """
    if traj is None:
        traj = []
    T = len(traj)
    if T == 0:
        return {"weights": [], "top_indices": [], "importance": []}

    speed = np.array([float(x.get('speed', 0.0)) for x in traj], dtype=np.float32)
    heading = np.array([float(x.get('heading', 0.0)) for x in traj], dtype=np.float32)
    altitude = np.array([float(x.get('altitude', 0.0)) for x in traj], dtype=np.float32)

    # time diffs (prepend 0 for t=0)
    d_speed = np.diff(speed, prepend=speed[0])
    d_heading = np.diff(heading, prepend=heading[0])
    d_heading = np.minimum(np.abs(d_heading), 360.0 - np.abs(d_heading))  # wrap-around aware
    d_alt = np.diff(altitude, prepend=altitude[0])

    # importance heuristic
    importance = alpha * np.abs(d_heading) + beta * np.abs(d_speed) + gamma * np.abs(d_alt)

    # softmax to [0,1]
    if importance.size == 0:
        return {"weights": [], "top_indices": [], "importance": []}
    imp_shift = importance - np.max(importance)
    weights = np.exp(imp_shift)
    denom = np.sum(weights)
    if denom <= 0:
        denom = 1.0
    weights = weights / denom

    # pick top-k indices
    k = int(min(top_k, T))
    top_indices = np.argsort(-weights)[:k]

    return {
        "weights": weights.tolist(),
        "top_indices": top_indices.tolist(),
        "importance": importance.tolist(),
    }


def build_attn_dsl_block(traj, attn):
    """
    Format an ATTN block for the DSL with top salient points.
    """
    if traj is None:
        traj = []
    lines = ["ATTN: top salient points by softmax weights"]
    top = attn.get("top_indices", [])
    weights = attn.get("weights", [])
    for idx in top:
        if idx < 0 or idx >= len(traj):
            continue
        x = traj[idx]
        w = weights[idx] if idx < len(weights) else 0.0
        ts = x.get("timestamp", idx)
        speed = float(x.get("speed", 0.0))
        heading = float(x.get("heading", 0.0))
        alt = float(x.get("altitude", 0.0))
        lines.append(
            f"- t={ts}, w={w:.3f}, speed={speed:.3f}, heading={heading:.1f}, alt={alt:.3f}"
        )
    return "\n".join(lines)