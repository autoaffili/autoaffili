# app.py
from fastapi import FastAPI
import threading
import whisper_worker

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/start")
def start():
    threading.Thread(target=whisper_worker.start_worker, daemon=True).start()
    return {"started": True}
