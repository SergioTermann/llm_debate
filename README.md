# Drone Swarm Evaluation via Multi‑Agent Debate

This repository implements a multi‑agent debate system to evaluate drone (UAV) swarm performance. It combines a scoring engine (flight control, swarm coordination, safety) with expert agents that debate and synthesize a final report. The system supports real API calls or a fully offline mock mode.

## Features
- Multi‑agent debate (3 expert roles: control, coordination, safety)
- Scoring engine with weighted categories and total score
- Robust trajectory parser for `trajectory.pkl` (nested dict/list/ndarray)
- Real model API mode and mock debate mode
- Structured outputs: flight data and evaluation report (JSON)
- Architecture diagram and LaTeX pseudocode for papers

## Project Layout
- `direct_debate.py` — Full evaluation pipeline (data → scoring → debate → report)
- `real_api_debate.py` — Minimal multi‑agent debate using real API
- `mock_debate.py` — Offline debate with handcrafted responses
- `trajectory.pkl` — Input trajectory data (see Data Input)
- `project_architecture_diagram.svg` — System overview diagram
- `algorithm_topconf.tex`, `algorithm_overleaf_minimal.tex`, `algorithm_topconf_math.tex` — LaTeX pseudocode (Overleaf‑ready)
- Outputs: `drone_flight_data.json`, `drone_swarm_evaluation_result.json`

## Requirements
- Python 3.9+
- Install deps: `pip install -r requirements.txt`

## Configuration
- API key: set `ZHIPU_API_KEY` (PowerShell: `$env:ZHIPU_API_KEY="YOUR_KEY"`)
- Optional proxies: set `http_proxy` / `https_proxy` if needed
- Optional base URL: `ZHIPU_BASE_URL` (use official domestic endpoint if applicable)

## Quick Start
1) Connectivity (optional): `python connect_test.py`
2) Real API debate:
   - Set `ZHIPU_API_KEY`
   - Run `python direct_debate.py` (full pipeline) or `python real_api_debate.py` (minimal)
3) Mock debate (offline): `python mock_debate.py`

## Data Input
- `trajectory.pkl` may contain nested structures. The parser in `direct_debate.py` searches for numeric arrays along paths like `trajectory/positions/observations/coords/path` and falls back to synthetic data if none are found.

## Outputs
- `drone_flight_data.json` — normalized flight data summary
- `drone_swarm_evaluation_result.json` — debate transcripts, expert roles, category scores, total score, recommendations and analysis

## Architecture & Pseudocode
- Diagram: `project_architecture_diagram.svg`
- Pseudocode (Overleaf‑ready):
  - `algorithm_topconf.tex` (top‑conf algorithmicx style)
  - `algorithm_overleaf_minimal.tex` (minimal algorithm2e)
  - `algorithm_topconf_math.tex` (math‑first, minimal text)

## Troubleshooting
- API errors: verify `ZHIPU_API_KEY`, base URL, network/proxy; update SDK; run `connect_test.py`.
- Trajectory parsing: ensure `trajectory.pkl` contains numeric arrays; the parser prints a notice and uses synthetic data if no valid array is found.

## Acknowledgements
Based on the idea of multi‑agent debate for improving reasoning. This implementation adapts it to UAV swarm evaluation.