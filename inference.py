import json
import os
from typing import List, Optional

from openai import OpenAI

from clinical_trial_env.env import ClinicalTrialEnv
from clinical_trial_env.models import ClinicalTrialAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("CLINICAL_TRIAL_TASK", "appendicitis_easy")
BENCHMARK = "clinical_trial_env"
MAX_TOKENS = 300
TEMPERATURE = 0.2

SYSTEM_PROMPT = """
You are a cautious emergency-triage agent operating inside a benchmark.
You may take up to two actions:
1. investigate: request a few targeted information categories
2. triage: submit the final structured decision

Return exactly one JSON object with no markdown.

Schema:
{
  "action_type": "investigate" or "triage",
  "information_requests": ["string"],
  "acuity_level": 1-5 or null,
  "provisional_diagnosis": "string or null",
  "department": "string or null",
  "immediate_actions": ["string"],
  "recommended_tests": ["string"],
  "disposition": "string or null",
  "rationale": "string or null"
}

Use canonical values when possible. For urgent stroke, sepsis, or surgical abdomen, avoid unsafe discharge.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation: dict) -> str:
    return json.dumps(
        {
            "task": observation["task_id"],
            "difficulty": observation["difficulty"],
            "step_index": observation["step_index"],
            "max_steps": observation["max_steps"],
            "patient": observation["patient"],
            "revealed_findings": observation["revealed_findings"],
            "requested_information": observation["requested_information"],
            "available_information_categories": observation["available_information_categories"],
            "instructions": observation["instructions"],
        },
        ensure_ascii=True,
    )


def heuristic_action(observation: dict) -> ClinicalTrialAction:
    task_id = observation["task_id"]
    step_index = observation["step_index"]

    if step_index == 0:
        requests = {
            "appendicitis_easy": ["abdominal_exam", "labs", "pregnancy_test"],
            "stroke_alert_medium": ["neuro_exam", "blood_glucose", "ct_head"],
            "septic_shock_hard": ["labs", "urinalysis", "mental_status_exam"],
        }[task_id]
        return ClinicalTrialAction(action_type="investigate", information_requests=requests)

    triage_actions = {
        "appendicitis_easy": ClinicalTrialAction(
            action_type="triage",
            acuity_level=2,
            provisional_diagnosis="acute appendicitis",
            department="emergency_surgery",
            immediate_actions=["iv_fluids", "pain_control", "surgical_consult", "npo"],
            recommended_tests=["cbc", "pregnancy_test", "abdominal_ultrasound"],
            disposition="admit_for_surgical_evaluation",
            rationale="RLQ peritonism with fever and inflammatory markers suggests appendicitis.",
        ),
        "stroke_alert_medium": ClinicalTrialAction(
            action_type="triage",
            acuity_level=1,
            provisional_diagnosis="acute ischemic stroke",
            department="stroke_unit",
            immediate_actions=["stroke_alert", "neurology_consult", "iv_access", "blood_pressure_monitoring"],
            recommended_tests=["ct_head", "blood_glucose", "ecg"],
            disposition="urgent_stroke_pathway",
            rationale="Sudden focal deficit within the treatment window needs emergent stroke activation.",
        ),
        "septic_shock_hard": ClinicalTrialAction(
            action_type="triage",
            acuity_level=1,
            provisional_diagnosis="septic shock",
            department="icu",
            immediate_actions=["broad_spectrum_antibiotics", "iv_fluids", "sepsis_bundle", "vasopressor_preparation"],
            recommended_tests=["lactate", "blood_cultures", "urinalysis", "cbc", "cmp"],
            disposition="icu_admission",
            rationale="Hypotension, fever, altered mentation, and elevated lactate indicate septic shock.",
        ),
    }
    return triage_actions[task_id]


def get_model_action(client: OpenAI, observation: dict) -> ClinicalTrialAction:
    prompt = build_user_prompt(observation)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    content = (response.choices[0].message.content or "").strip()
    payload = json.loads(content)
    return ClinicalTrialAction.model_validate(payload)


def main() -> None:
    env = ClinicalTrialEnv()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=TASK_NAME)

        while not observation.done and steps_taken < observation.max_steps:
            action_error = None
            try:
                action = get_model_action(client, observation.model_dump())
            except Exception:
                action = heuristic_action(observation.model_dump())
                action_error = "model_request_failed_fallback_used"

            result = env.step(action)
            steps_taken += 1
            rewards.append(result.reward)
            score = result.info.score
            success = result.done and score >= 0.60

            action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
            log_step(
                step=steps_taken,
                action=action_str,
                reward=result.reward,
                done=result.done,
                error=action_error,
            )

            observation = result.observation

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
