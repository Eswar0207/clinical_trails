from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PatientSnapshot(BaseModel):
    age: int
    sex: str
    chief_complaint: str
    vitals: Dict[str, str]
    known_history: List[str] = Field(default_factory=list)
    context: str


class ClinicalTrialAction(BaseModel):
    action_type: Literal["investigate", "triage"] = "triage"
    information_requests: List[str] = Field(default_factory=list)
    acuity_level: Optional[int] = None
    provisional_diagnosis: Optional[str] = None
    department: Optional[str] = None
    immediate_actions: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    disposition: Optional[str] = None
    rationale: Optional[str] = None


class ClinicalTrialObservation(BaseModel):
    benchmark: str = "clinical_trial_env"
    task_id: str
    difficulty: str
    step_index: int
    max_steps: int
    patient: PatientSnapshot
    revealed_findings: Dict[str, str] = Field(default_factory=dict)
    requested_information: List[str] = Field(default_factory=list)
    available_information_categories: List[str] = Field(default_factory=list)
    instructions: str
    last_reward: float = 0.0
    done: bool = False


class ClinicalTrialState(BaseModel):
    task_id: str
    step_index: int
    max_steps: int
    requested_information: List[str] = Field(default_factory=list)
    revealed_findings: Dict[str, str] = Field(default_factory=dict)
    reward_history: List[float] = Field(default_factory=list)
    done: bool = False


class ClinicalTrialInfo(BaseModel):
    score: float
    phase: Literal["investigation", "triage", "finished"]
    evaluation: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: ClinicalTrialObservation
    reward: float
    done: bool
    info: ClinicalTrialInfo


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StateResponse(BaseModel):
    state: ClinicalTrialState


class HealthResponse(BaseModel):
    status: str
    benchmark: str
    tasks: List[str]
