---
title: Clinical Trails
emoji: 🦀
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
license: mit
---

# ClinicalTrialEnv

ClinicalTrialEnv is a real-world OpenEnv-style benchmark for emergency-room triage. An agent receives a patient case, optionally requests targeted information, and then submits a triage decision covering urgency, diagnosis, tests, destination, and immediate interventions.

The environment is designed for the OpenEnv hackathon rubric:

- Real-world task: emergency intake and triage, not a toy game
- Typed observation, action, reward, and state models
- `reset()` / `step()` / `state()` API
- Three deterministic tasks with easy, medium, and hard difficulty
- Reward shaping with partial credit for near-correct decisions
- Root-level `inference.py` using the OpenAI Python client
- Dockerized FastAPI app suitable for Hugging Face Spaces

## Why this is a strong submission

This project models a meaningful human workflow: deciding which patients need immediate escalation, which department should receive them, what actions are time-critical, and when discharge is unsafe. It is useful for evaluating planning, risk management, and structured decision-making in LLM agents.

## Environment summary

Each episode has up to 2 steps:

1. Investigation step
   The agent requests focused information such as `neuro_exam`, `pregnancy_test`, `ecg`, or `labs`.
   The environment returns hidden findings relevant to the case and gives a small shaped reward for useful requests.

2. Triage step
   The agent submits a final triage plan:
   - `acuity_level` from 1 to 5
   - `provisional_diagnosis`
   - `department`
   - `immediate_actions`
   - `recommended_tests`
   - `disposition`
   - `rationale`

The episode ends after the triage step or when the maximum step count is reached.

## Tasks

The benchmark ships with three tasks.

### 1. `appendicitis_easy`
- Difficulty: easy
- Scenario: probable appendicitis with classic right-lower-quadrant pain
- Main challenge: avoid under-triage and choose surgery/ED workflow correctly

### 2. `stroke_alert_medium`
- Difficulty: medium
- Scenario: acute stroke symptoms with time-sensitive red flags
- Main challenge: route urgently, request the right diagnostic workup, avoid unsafe delay

### 3. `septic_shock_hard`
- Difficulty: hard
- Scenario: sepsis with hypotension, fever, and organ dysfunction risk
- Main challenge: recognize instability, choose ICU-level escalation, and prioritize life-saving actions

## Reward design

Scores are deterministic and normalized to `[0.0, 1.0]`.

- Investigation reward: up to `0.20`
- Final triage reward: up to `0.80`
- Safety penalties apply for dangerous decisions such as major under-triage or discharging an unstable patient

Final reward components include:

- acuity correctness
- diagnosis match using deterministic keyword aliases
- department correctness
- overlap with required immediate actions
- overlap with recommended tests
- correct disposition
- safety/risk penalties

## Action and observation spaces

### Observation

The observation is a typed JSON object containing:

- task metadata
- patient vignette and vitals
- revealed findings
- requested information so far
- allowed information categories
- episode progress

### Action

The action is a typed JSON object with:

- `action_type`: `investigate` or `triage`
- `information_requests`: list of targeted data requests for investigation
- triage fields for final decision submission

## Files

- [inference.py](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/inference.py): baseline agent runner
- [openenv.yaml](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/openenv.yaml): environment metadata
- [Dockerfile](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/Dockerfile): container build
- [app.py](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/app.py): FastAPI entrypoint for the Space
- [clinical_trial_env/env.py](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/clinical_trial_env/env.py): environment logic
- [clinical_trial_env/tasks.py](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/clinical_trial_env/tasks.py): benchmark task definitions

## Setup

```bash
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
```

## Environment variables for `inference.py`

Define these before running:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME`

Defaults are only provided for:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`

## Run baseline inference

```bash
python inference.py
```

Optional task selection:

```bash
set CLINICAL_TRIAL_TASK=stroke_alert_medium
python inference.py
```

## OpenEnv notes

This repo follows the requested hackathon structure and OpenEnv-style interface. If the official validator expects an adapter class from a specific OpenEnv package version, the environment logic in [clinical_trial_env/env.py](/C:/Users/HP/OneDrive/Pictures/Documents/New%20project/clinical_trial_env/env.py) is intentionally isolated so that adapter wiring can be added with minimal changes.
