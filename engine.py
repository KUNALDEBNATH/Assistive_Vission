import csv
import os
import tempfile
import time
import threading

import cv2
import easyocr
from gtts import gTTS
from playsound import playsound
from ultralytics import YOLO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration 

_last_message_lock = threading.Lock()
_last_message = "No message yet."

def set_last_message(text: str):
    global _last_message
    with _last_message_lock:
        _last_message = text or ""

def get_last_message() -> str:
    with _last_message_lock:
        return _last_message or "No message yet."

def speak(text: str, lang: str = "en"):
    if not text:
        return

    set_last_message(text)
    print("[TTS]", text)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        path = fp.name
    tts = gTTS(text=text, lang=lang)
    tts.save(path)
    playsound(path)
    os.remove(path)


# -------------------- GEOMETRY & OCR --------------------

def bbox_center_and_area(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    area = float(max(w, 0) * max(h, 0))
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, area


def direction_from_center(cx, frame_width, dead_zone_ratio=0.25):
    center = frame_width / 2.0
    dead_zone = dead_zone_ratio * frame_width / 2.0
    if cx < center - dead_zone:
        return "left"
    elif cx > center + dead_zone:
        return "right"
    else:
        return "ahead"


def estimate_distance_from_area(area, frame_area):
    if area <= 0:
        return None
    ratio = area / max(frame_area, 1.0)
    if ratio > 0.4:
        return 0.5
    elif ratio > 0.2:
        return 1.0
    elif ratio > 0.1:
        return 1.5
    elif ratio > 0.05:
        return 2.0
    elif ratio > 0.02:
        return 3.0
    else:
        return 4.0


def run_ocr_and_speak(reader: easyocr.Reader, frame):
    result = reader.readtext(frame, detail=0, paragraph=True)
    if not result:
        print("[OCR] No readable text.")
        return "ocr_none"
    text = " ".join(result)
    print("[OCR TEXT]", text)
    speak(text)
    return text


def init_logger(path: str):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "timestamp", "mode", "target_class",
            "focus_label", "focus_zone",
            "direction", "distance_m",
            "event_type", "spoken_text"
        ])
    return f, writer


# -------------------- LOCAL BLIP CAPTIONER --------------------

class LocalCaptioner:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading BLIP captioning model on", self.device)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )  # [web:9]
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(self, frame_bgr):
        image_rgb = frame_bgr[:, :, ::-1].copy()
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3
        )
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()


# -------------------- ASSISTIVE ENGINE CLASS --------------------

class AssistiveEngine:
    """
    Assistive YOLOv12 + EasyOCR + BLIP + gTTS engine.
    """

    MODEL_NAME = "yolo12n.pt"
    CONF_THRESHOLD = 0.35  # slightly lower to detect more obstacles

    MODE = "navigation"      # "navigation" or "explore"
    TARGET_CLASS = None

    NEAR_THRESHOLD = 1.0
    MID_THRESHOLD = 2.0

    OCR_INTERVAL = 10.0
    VLM_INTERVAL = 15.0

    LOG_PATH = "logs.csv"

    OBSTACLE_CLASSES = {
        "person", "bicycle", "car", "motorbike", "bus", "truck",
        "traffic light", "stop sign", "bench", "chair", "table", "sofa", "bed",
    }  # COCO-style obstacle set[web:58][web:60]

    def __init__(
        self,
        model_path: str = None,
        mode: str = None,
        target_class: str = None,
        ocr_interval: float = None,
        vlm_interval: float = None,
        log_path: str = None,
        camera_index: int = 0,
    ):
        self.model_path = model_path or self.MODEL_NAME
        self.mode = mode or self.MODE
        self.target_class = target_class or self.TARGET_CLASS
        self.ocr_interval = self.OCR_INTERVAL if ocr_interval is None else ocr_interval
        self.vlm_interval = self.VLM_INTERVAL if vlm_interval is None else vlm_interval
        self.log_path = log_path or self.LOG_PATH
        self.camera_index = camera_index

        self._stop_flag = False

        self.model = None
        self.ocr_reader = None
        self.captioner = None

    def get_latest_message(self) -> str:
        return get_last_message()

    def stop(self):
        self._stop_flag = True

    def _lazy_init(self):
        if self.model is None:
            print(f"Loading YOLO model {self.model_path} ...")
            self.model = YOLO(self.model_path)  # [web:2]

        if self.ocr_reader is None:
            print("Loading OCR model (EasyOCR)...")
            self.ocr_reader = easyocr.Reader(["en"])

        if self.captioner is None:
            print("Initializing local BLIP captioner...")
            self.captioner = LocalCaptioner()

    def run(self):
        self._lazy_init()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        log_file, log_writer = init_logger(self.log_path)

        print(
            f"Mode={self.mode}, Target={self.target_class}, "
            f"OCR_INTERVAL={self.ocr_interval}, VLM_INTERVAL={self.vlm_interval}"
        )
        print("Assistive loop (no GUI).")

        last_ocr_time = time.time()
        last_vlm_time = time.time()

        focus_label = None
        focus_zone = None
        focus_last_seen = 0.0

        try:
            while not self._stop_flag:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from webcam.")
                    break

                h, w, _ = frame.shape
                frame_area = float(w * h)

                results = self.model.predict(
                    source=frame,
                    conf=self.CONF_THRESHOLD,
                    verbose=False,
                )
                r = results[0]
                boxes = r.boxes
                names = self.model.names

                best_obj = None  # (label, conf, cx, cy, area, score)

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = names[cls_id]

                        cx, cy, area = bbox_center_and_area(x1, y1, x2, y2)
                        if area <= 0:
                            continue

                        direction = direction_from_center(cx, w)
                        score = area

                        # Center gets priority
                        if direction == "ahead":
                            score *= 2.0

                        # Known obstacles get extra weight
                        if self.mode == "navigation" and label in self.OBSTACLE_CLASSES:
                            score *= 1.5

                        # Target class strongest
                        if self.target_class is not None and label == self.target_class:
                            score *= 4.0

                        if best_obj is None or score > best_obj[-1]:
                            best_obj = (label, conf, cx, cy, area, score)

                now = time.time()

                # -------- Event-based speech --------
                if best_obj is not None:
                    label, conf, cx, cy, area, score = best_obj
                    direction = direction_from_center(cx, w)
                    dist_m = estimate_distance_from_area(area, frame_area)

                    if dist_m is None:
                        zone = None
                    elif dist_m < self.NEAR_THRESHOLD:
                        zone = "near"
                    elif dist_m < self.MID_THRESHOLD:
                        zone = "mid"
                    else:
                        zone = "far"

                    should_speak = False
                    event_type = ""
                    sentence = ""

                    if self.target_class is not None and label == self.target_class:
                        if focus_label != self.target_class:
                            sentence = f"{self.target_class} {direction}, about {dist_m:.1f} meters. Target found."
                            event_type = "target_appeared"
                            should_speak = True
                        else:
                            if zone != focus_zone and zone in ("mid", "near"):
                                if zone == "near":
                                    sentence = f"{self.target_class} {direction}, within one meter. You are very close."
                                    event_type = "target_near"
                                else:
                                    sentence = f"{self.target_class} {direction}, about {dist_m:.1f} meters."
                                    event_type = "target_closer"
                                should_speak = True
                    else:
                        if focus_label is None:
                            if zone == "near":
                                sentence = f"{label} {direction}, within one meter."
                            elif zone == "mid":
                                sentence = f"{label} {direction}, about {dist_m:.1f} meters."
                            else:
                                sentence = f"{label} {direction}, ahead."
                            event_type = "first_obstacle"
                            should_speak = True
                        else:
                            if label != focus_label:
                                if zone == "near":
                                    sentence = f"Now {label} {direction}, within one meter."
                                elif zone == "mid":
                                    sentence = f"Now {label} {direction}, about {dist_m:.1f} meters."
                                else:
                                    sentence = f"{label} {direction}, ahead."
                                event_type = "obstacle_changed"
                                should_speak = True
                            else:
                                if zone != focus_zone and zone in ("mid", "near"):
                                    if zone == "near":
                                        sentence = f"{label} {direction}, very close."
                                        event_type = "obstacle_near"
                                    else:
                                        sentence = f"{label} {direction}, about {dist_m:.1f} meters."
                                        event_type = "obstacle_closer"
                                    should_speak = True

                    if should_speak and sentence:
                        speak(sentence)
                        log_writer.writerow([
                            now, self.mode, self.target_class or "",
                            label, zone or "",
                            direction, dist_m if dist_m is not None else "",
                            event_type, sentence
                        ])
                        log_file.flush()

                        focus_label = (
                            label if self.target_class is None
                            else (self.target_class if label == self.target_class else label)
                        )
                        focus_zone = zone
                        focus_last_seen = now
                    else:
                        focus_label = label
                        focus_zone = zone
                        focus_last_seen = now
                else:
                    # No detections: maybe clear, maybe wall
                    if focus_label is not None and now - focus_last_seen > 1.0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        var = gray.var()

                        if var < 200:  # heuristic, tune if needed
                            sentence = "Solid surface ahead, possibly a wall, very close."
                            event_type = "possible_wall"
                        else:
                            sentence = "Path ahead seems clear."
                            event_type = "path_clear"

                        speak(sentence)
                        log_writer.writerow([
                            now, self.mode, self.target_class or "",
                            "", "", "", "",
                            event_type, sentence
                        ])
                        log_file.flush()
                        focus_label = None
                        focus_zone = None
                        focus_last_seen = now

                # -------- OCR events --------
                if self.ocr_interval is not None and now - last_ocr_time >= self.ocr_interval:
                    print("[OCR] Running OCR on current frame...")
                    text = run_ocr_and_speak(self.ocr_reader, frame)
                    log_writer.writerow([
                        now, self.mode, self.target_class or "",
                        "", "", "", "",
                        "ocr", text
                    ])
                    log_file.flush()
                    last_ocr_time = now

                # -------- BLIP caption events --------
                if self.vlm_interval is not None and now - last_vlm_time >= self.vlm_interval:
                    print("[CAPTION] Generating BLIP caption for current scene...")
                    caption = self.captioner.caption(frame)
                    if caption:
                        sentence = f"Scene summary: {caption}"
                        speak(sentence)
                        log_writer.writerow([
                            now, self.mode, self.target_class or "",
                            "", "", "", "",
                            "vlm_caption", sentence
                        ])
                        log_file.flush()
                    last_vlm_time = now

        except KeyboardInterrupt:
            print("Stopping (KeyboardInterrupt)...")

        finally:
            cap.release()
            log_file.close()
            print("AssistiveEngine stopped cleanly.")


if __name__ == "__main__":
    engine = AssistiveEngine()
    engine.run()
