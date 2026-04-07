from fastapi import FastAPI

from clinical_trial_env.server import create_app

app: FastAPI = create_app()
