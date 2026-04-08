from fastapi import FastAPI
import uvicorn

from clinical_trial_env.server import create_app

app: FastAPI = create_app()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
