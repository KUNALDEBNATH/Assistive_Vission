import time
import threading

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from engine import AssistiveEngine


app = FastAPI()
templates = Jinja2Templates(directory="templates")

assistive_engine = AssistiveEngine(camera_index=0)


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam in server.py")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam in server.py.")
                break

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            jpg_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        cap.release()


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=assistive_engine.run, daemon=True)
    thread.start()
    print("AssistiveEngine background thread started.")


@app.on_event("shutdown")
def shutdown_event():
    assistive_engine.stop()
    print("AssistiveEngine stop requested.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video")
async def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
async def status():
    msg = assistive_engine.get_latest_message()
    return JSONResponse({"message": msg})
