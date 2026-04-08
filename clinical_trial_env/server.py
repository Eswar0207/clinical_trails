from __future__ import annotations

from fastapi import FastAPI

from .env import ClinicalTrialEnv
from .models import (
    ClinicalTrialAction,
    ClinicalTrialObservation,
    HealthResponse,
    ResetRequest,
    StateResponse,
    StepResult,
    TaskDescriptor,
    TasksResponse,
)
from .tasks import TASKS


def create_app() -> FastAPI:
    app = FastAPI(title="ClinicalTrialEnv", version="0.1.0")
    env = ClinicalTrialEnv()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", benchmark="clinical_trial_env", tasks=list(TASKS.keys()))

    @app.post("/reset", response_model=ClinicalTrialObservation)
    def reset(request: ResetRequest | None = None) -> ClinicalTrialObservation:
        task_id = request.task_id if request else None
        return env.reset(task_id=task_id)

    @app.post("/step", response_model=StepResult)
    def step(action: ClinicalTrialAction) -> StepResult:
        return env.step(action)

    @app.get("/state", response_model=StateResponse)
    def state() -> StateResponse:
        return StateResponse(state=env.state())

    @app.get("/tasks", response_model=TasksResponse)
    def tasks() -> TasksResponse:
        return TasksResponse(
            tasks=[
                TaskDescriptor(id=task_id, difficulty=task.difficulty, grader_enabled=True)
                for task_id, task in TASKS.items()
            ]
        )

    return app
