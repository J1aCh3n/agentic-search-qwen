"""Application entry point."""
from fastapi import FastAPI

app = FastAPI(title="CloudEngine API")

@app.get("/health")
def health_check():
    return {"status": "ok"}
