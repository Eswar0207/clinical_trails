from __future__ import annotations

from typing import Dict, List, Tuple

from .models import ClinicalTrialAction, ClinicalTrialInfo, ClinicalTrialObservation, ClinicalTrialState, StepResult
from .tasks import TASKS, TaskSpec


BENCHMARK_NAME = "clinical_trial_env"
MAX_STEPS = 2


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())


def _overlap_score(predicted: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    predicted_set = {_normalize_text(item) for item in predicted}
    expected_set = {_normalize_text(item) for item in expected}
    hits = len(predicted_set & expected_set)
    return hits / len(expected_set)


class ClinicalTrialEnv:
    def __init__(self) -> None:
        self._task: TaskSpec | None = None
        self._step_index = 0
        self._requested_information: List[str] = []
        self._revealed_findings: Dict[str, str] = {}
        self._reward_history: List[float] = []
        self._done = False

    def reset(self, task_id: str | None = None) -> ClinicalTrialObservation:
        selected_task = TASKS[task_id] if task_id else next(iter(TASKS.values()))
        self._task = selected_task
        self._step_index = 0
        self._requested_information = []
        self._revealed_findings = {}
        self._reward_history = []
        self._done = False
        return self._build_observation(last_reward=0.0)

    def step(self, action: ClinicalTrialAction) -> StepResult:
        if self._task is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._done:
            raise RuntimeError("Episode already completed.")

        self._step_index += 1
        if action.action_type == "investigate" and self._step_index < MAX_STEPS:
            reward, notes = self._handle_investigation(action)
            self._reward_history.append(reward)
            observation = self._build_observation(last_reward=reward)
            info = ClinicalTrialInfo(
                score=self.current_score(),
                phase="investigation",
                evaluation={"investigation_reward": reward},
                notes=notes,
            )
            return StepResult(observation=observation, reward=reward, done=False, info=info)

        reward, evaluation, notes = self._handle_triage(action)
        self._reward_history.append(reward)
        self._done = True
        observation = self._build_observation(last_reward=reward, done=True)
        info = ClinicalTrialInfo(
            score=self.current_score(),
            phase="finished",
            evaluation=evaluation,
            notes=notes,
        )
        return StepResult(observation=observation, reward=reward, done=True, info=info)

    def state(self) -> ClinicalTrialState:
        if self._task is None:
            raise RuntimeError("Environment must be reset before calling state().")
        return ClinicalTrialState(
            task_id=self._task.task_id,
            step_index=self._step_index,
            max_steps=MAX_STEPS,
            requested_information=list(self._requested_information),
            revealed_findings=dict(self._revealed_findings),
            reward_history=list(self._reward_history),
            done=self._done,
        )

    def close(self) -> None:
        return None

    def current_score(self) -> float:
        score = sum(self._reward_history)
        return min(max(score, 0.0), 1.0)

    def _build_observation(self, last_reward: float, done: bool = False) -> ClinicalTrialObservation:
        assert self._task is not None
        return ClinicalTrialObservation(
            benchmark=BENCHMARK_NAME,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_index=self._step_index,
            max_steps=MAX_STEPS,
            patient=self._task.patient,
            revealed_findings=dict(self._revealed_findings),
            requested_information=list(self._requested_information),
            available_information_categories=list(self._task.hidden_findings.keys()),
            instructions=(
                "Use one investigation step if helpful, then submit a final triage decision. "
                "Rewards favor safe acuity, correct routing, and clinically appropriate actions."
            ),
            last_reward=last_reward,
            done=done,
        )

    def _handle_investigation(self, action: ClinicalTrialAction) -> Tuple[float, List[str]]:
        assert self._task is not None
        requested = [_normalize_text(item) for item in action.information_requests]
        notes: List[str] = []
        reward = 0.0

        if not requested:
            notes.append("No targeted information was requested.")
            return 0.0, notes

        for raw_request in requested:
            if raw_request in self._requested_information:
                reward -= 0.02
                notes.append(f"Repeated request: {raw_request}")
                continue

            self._requested_information.append(raw_request)

            matched_key = next(
                (key for key in self._task.hidden_findings.keys() if _normalize_text(key) == raw_request),
                None,
            )
            if matched_key is None:
                reward -= 0.01
                notes.append(f"Irrelevant request: {raw_request}")
                continue

            self._revealed_findings[matched_key] = self._task.hidden_findings[matched_key]
            if matched_key in self._task.useful_information:
                reward += 0.05
                notes.append(f"Useful information revealed: {matched_key}")
            else:
                reward += 0.01
                notes.append(f"Information revealed: {matched_key}")

        return min(max(reward, 0.0), 0.20), notes

    def _handle_triage(self, action: ClinicalTrialAction) -> Tuple[float, Dict[str, float], List[str]]:
        assert self._task is not None
        notes: List[str] = []

        acuity_score = self._score_acuity(action.acuity_level)
        diagnosis_score = self._score_diagnosis(action.provisional_diagnosis)
        department_score = self._score_exact(action.department, self._task.target_department)
        action_score = _overlap_score(action.immediate_actions, self._task.target_immediate_actions)
        tests_score = _overlap_score(action.recommended_tests, self._task.target_tests)
        disposition_score = self._score_exact(action.disposition, self._task.target_disposition)
        investigation_bonus = min(len(self._revealed_findings), len(self._task.useful_information)) / max(
            len(self._task.useful_information), 1
        )

        base_reward = (
            0.20 * acuity_score
            + 0.20 * diagnosis_score
            + 0.10 * department_score
            + 0.15 * action_score
            + 0.10 * tests_score
            + 0.15 * disposition_score
            + 0.10 * investigation_bonus
        )

        safety_penalty = self._safety_penalty(action)
        final_reward = min(max(base_reward - safety_penalty, 0.0), 0.80)

        if safety_penalty > 0:
            notes.append(f"Safety penalty applied: -{safety_penalty:.2f}")

        evaluation = {
            "acuity": round(acuity_score, 4),
            "diagnosis": round(diagnosis_score, 4),
            "department": round(department_score, 4),
            "immediate_actions": round(action_score, 4),
            "recommended_tests": round(tests_score, 4),
            "disposition": round(disposition_score, 4),
            "investigation_bonus": round(investigation_bonus, 4),
            "safety_penalty": round(safety_penalty, 4),
            "final_reward": round(final_reward, 4),
        }
        return final_reward, evaluation, notes

    def _score_acuity(self, acuity: int | None) -> float:
        if acuity is None:
            return 0.0
        delta = abs(acuity - self._task.target_acuity)
        if delta == 0:
            return 1.0
        if delta == 1:
            return 0.6
        if delta == 2:
            return 0.2
        return 0.0

    def _score_diagnosis(self, diagnosis: str | None) -> float:
        normalized = _normalize_text(diagnosis)
        if not normalized:
            return 0.0
        aliases = [_normalize_text(item) for item in self._task.acceptable_diagnosis_aliases]
        if normalized in aliases:
            return 1.0
        if any(alias in normalized or normalized in alias for alias in aliases):
            return 0.6
        return 0.0

    def _score_exact(self, actual: str | None, expected: str) -> float:
        return 1.0 if _normalize_text(actual) == _normalize_text(expected) else 0.0

    def _safety_penalty(self, action: ClinicalTrialAction) -> float:
        penalty = 0.0
        acuity = action.acuity_level
        disposition = _normalize_text(action.disposition)

        if acuity is None:
            penalty += 0.15
        elif acuity - self._task.target_acuity >= 2:
            penalty += 0.25

        if disposition in {_normalize_text(item) for item in self._task.unsafe_dispositions}:
            penalty += 0.30

        return min(penalty, 0.50)
