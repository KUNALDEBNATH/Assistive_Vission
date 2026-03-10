# 🦯 Assistive Vision

A real-time assistive system for visually impaired users that uses a webcam to detect obstacles, read text, and describe scenes — all via spoken audio feedback. Built with YOLOv12, EasyOCR, BLIP, and gTTS, served through a FastAPI web interface.

---

## ✨ Features

- **Obstacle Detection** — Detects people, vehicles, furniture, and other hazards using YOLOv12. Announces their direction (left, ahead, right) and estimated distance in meters.
- **Text Recognition (OCR)** — Periodically reads visible text in the scene aloud using EasyOCR.
- **Scene Captioning (VLM)** — Every 15 seconds, generates a natural language scene summary using the BLIP image captioning model (runs fully locally).
- **Text-to-Speech** — All messages are spoken aloud via Google TTS (gTTS).
- **Live Web UI** — A browser-based dashboard streams the webcam feed and shows the latest spoken message in real time.
- **Event Logging** — Every detection event, OCR result, and caption is logged to `logs.csv` for review.
- **Wall Detection** — When no objects are detected and the image variance is low, the system warns about a possible wall or solid surface ahead.

---

## 🗂️ Project Structure

```
assistive-vision/
├── engine.py          # Core AI engine (YOLO + OCR + BLIP + TTS logic)
├── server.py          # FastAPI web server (video stream + status endpoint)
├── templates/
│   └── index.html     # Browser UI (live video + status panel)
├── yolo12n.pt         # YOLOv12 nano model weights
├── logs.csv           # Auto-generated event log
└── requirements.txt   # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A webcam
- CUDA-capable GPU *(optional but recommended for BLIP)*

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/assistive-vision.git
   cd assistive-vision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model weights** — ensure `yolo12n.pt` is in the project root directory.

### Running

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open your browser and navigate to:
```
http://localhost:8000
```

You will see the live webcam feed and a status panel showing the latest spoken message.

---

## ⚙️ Configuration

All tunable parameters are defined as class-level constants in `AssistiveEngine` inside `engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODE` | `"navigation"` | `"navigation"` (obstacle avoidance) or `"explore"` (general awareness) |
| `TARGET_CLASS` | `None` | If set (e.g. `"chair"`), the system prioritizes finding that object |
| `CONF_THRESHOLD` | `0.35` | YOLO detection confidence threshold |
| `NEAR_THRESHOLD` | `1.0 m` | Distance considered "near" |
| `MID_THRESHOLD` | `2.0 m` | Distance considered "mid-range" |
| `OCR_INTERVAL` | `10.0 s` | How often OCR runs on the current frame |
| `VLM_INTERVAL` | `15.0 s` | How often BLIP generates a scene caption |
| `LOG_PATH` | `"logs.csv"` | Path for the event log CSV file |

You can also pass these as constructor arguments:

```python
engine = AssistiveEngine(
    mode="navigation",
    target_class="chair",
    ocr_interval=20.0,
    camera_index=0
)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/video` | MJPEG webcam stream |
| `GET` | `/status` | Returns the latest spoken message as JSON: `{"message": "..."}` |

---

## 📋 Log Format

Events are written to `logs.csv` with the following columns:

| Column | Description |
|---|---|
| `timestamp` | Unix timestamp of the event |
| `mode` | Engine mode (`navigation` / `explore`) |
| `target_class` | The configured target class (if any) |
| `focus_label` | Detected object label |
| `focus_zone` | Proximity zone (`near`, `mid`, `far`) |
| `direction` | Spatial direction (`left`, `ahead`, `right`) |
| `distance_m` | Estimated distance in meters |
| `event_type` | Event category (e.g. `first_obstacle`, `target_found`, `ocr`, `vlm_caption`) |
| `spoken_text` | The exact text spoken aloud |

---

## 🧠 How It Works

1. On startup, the `AssistiveEngine` loads YOLOv12, EasyOCR, and BLIP in a background thread.
2. Each frame is run through YOLO. Detections are scored by size, position (center = higher priority), obstacle type, and whether they match the target class.
3. Speech is **event-driven** — the engine only speaks when something meaningful changes (new object appears, object gets closer, path clears, etc.) to avoid spamming the user.
4. OCR and BLIP captions fire on independent timers.
5. The FastAPI server independently streams raw webcam frames to the browser and exposes the latest message over `/status`, polled every 500 ms by the frontend.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv12 object detection |
| `easyocr` | On-frame text recognition |
| `transformers` | BLIP image captioning model |
| `gtts` | Google Text-to-Speech |
| `fastapi` + `uvicorn` | Web server |
| `opencv-python` | Webcam capture and image processing |
| `torch` | GPU inference backend |

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.
