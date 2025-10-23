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
- `pip` installed and up‑to‑date
- Supported OS: Windows 10/11, macOS 12+, Ubuntu 20.04+ (or compatible Linux)
- Optional Internet connectivity (required for real API mode; mock mode is offline)

## Installation
It is recommended to use a virtual environment.

- Windows (PowerShell):
  - `python -V`
  - `py -3.9 -m venv .venv`
  - `\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`

- macOS/Linux (bash/zsh):
  - `python3 -V`
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

## Configuration
Environment variables control API access and connectivity. Set them in your shell before running.

- API key:
  - Windows PowerShell: ``$env:ZHIPU_API_KEY="YOUR_KEY"``
  - macOS/Linux: ``export ZHIPU_API_KEY="YOUR_KEY"``

- Optional HTTP(S) proxy (e.g., local proxy at `http://localhost:7890`):
  - Windows PowerShell: ``$env:http_proxy="http://localhost:7890"`` and ``$env:https_proxy="http://localhost:7890"``
  - macOS/Linux: ``export http_proxy="http://localhost:7890"`` and ``export https_proxy="http://localhost:7890"``

- Optional base URL (if a custom or domestic endpoint is required):
  - Windows PowerShell: ``$env:ZHIPU_BASE_URL="https://your-endpoint"``
  - macOS/Linux: ``export ZHIPU_BASE_URL="https://your-endpoint"``

Notes:
- `direct_debate.py` sets `http_proxy`/`https_proxy` to `http://localhost:7890` at the top. If you do not use a local proxy, comment out or change these lines, or override via environment variables as shown above.
- Keep your API key secure. Do not hardcode it in source files or commit it to version control.

## Quick Start
1) Connectivity (optional): `python connect_test.py`
2) Real API debate:
   - Set `ZHIPU_API_KEY`
   - Run `python direct_debate.py` (full pipeline) or `python real_api_debate.py` (minimal)
3) Mock debate (offline): `python mock_debate.py`

## Run & Verify
- Verify environment:
  - `python -V` (or `python3 -V`) shows 3.9+
  - `pip -V` shows a recent pip version
- Connectivity test: `python connect_test.py`
- Full pipeline: `python direct_debate.py`
  - Produces `drone_flight_data.json` and `drone_swarm_evaluation_result.json` in the project root
- Minimal API example: `python real_api_debate.py`
- Offline mock: `python mock_debate.py`

## Data Input
- `trajectory.pkl` may contain nested structures. The parser in `direct_debate.py` searches for numeric arrays along paths like `trajectory/positions/observations/coords/path` and falls back to synthetic data if none are found.

## Outputs
- `drone_flight_data.json` — normalized flight data summary
- `drone_swarm_evaluation_result.json` — debate transcripts, expert roles, category scores, total score, recommendations and analysis

## Troubleshooting
- API errors:
  - Ensure `ZHIPU_API_KEY` is set and valid
  - If using a proxy, verify `http_proxy`/`https_proxy` and network connectivity
  - If a custom base URL is needed, set `ZHIPU_BASE_URL`
  - Run `python connect_test.py` to validate connectivity
- Dependency issues:
  - `ModuleNotFoundError`: re‑run `pip install -r requirements.txt`
  - Update `pip`: `python -m pip install --upgrade pip`
- Proxy conflicts:
  - If you do not use a local proxy, remove or edit the proxy lines at the top of `direct_debate.py`
- Trajectory parsing:
  - Ensure `trajectory.pkl` contains numeric arrays; the parser prints a notice and uses synthetic data if none are found

## Reproducibility
- Mock mode (`python mock_debate.py`) runs fully offline and is deterministic given fixed inputs
- For experiments, log seeds and environment variables; keep consistent weights in scoring configuration

## Architecture & Pseudocode
- Diagram: `project_architecture_diagram.svg`
- Pseudocode (Overleaf‑ready):
  - `algorithm_topconf.tex` (top‑conf algorithmicx style)
  - `algorithm_overleaf_minimal.tex` (minimal algorithm2e)
  - `algorithm_topconf_math.tex` (math‑first, minimal text)

## Acknowledgements
Based on the idea of multi‑agent debate for improving reasoning. This implementation adapts it to UAV swarm evaluation.