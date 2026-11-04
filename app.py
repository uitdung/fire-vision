import json
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" if (ROOT / "models").is_dir() else ROOT / "fire-detection-pi4" / "models"
MODEL_PATH = MODEL_ROOT / "mobilenetv3s_224_fp16.tflite"
MODEL_META_PATH = MODEL_ROOT / "mobilenetv3s_224.json"
YOLO_PATH = MODEL_ROOT / "yolov8n_416_int8.tflite"
YOLO_META_PATH = MODEL_ROOT / "yolov8n_416.json"
ALLOWED = {"png", "jpg", "jpeg", "webp", "mp4", "avi", "mov", "mkv", "webm"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
GATE_THRESHOLD = 0.5
FIRE_THRESHOLD = 0.15
SMOKE_THRESHOLD = 0.4
YOLO_SIZE = 416
YOLO_EVERY_SECONDS = 1.0
GATE_HOLD_SECONDS = 2.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
if not YOLO_PATH.is_file():
    raise FileNotFoundError(f"Missing model file: {YOLO_PATH}")

model_meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
input_size = int(model_meta["input_size"])
fire_index = int(model_meta["fire_index"])
yolo_meta = json.loads(YOLO_META_PATH.read_text(encoding="utf-8"))
YOLO_SIZE = int(yolo_meta["input_size"])
FIRE_THRESHOLD = float(yolo_meta["fire_conf_by_format"]["tflite_int8"])
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=2)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
yolo_interpreter = tf.lite.Interpreter(model_path=str(YOLO_PATH), num_threads=2)
yolo_interpreter.allocate_tensors()
yolo_input_info = yolo_interpreter.get_input_details()[0]
yolo_output_info = yolo_interpreter.get_output_details()[0]
inference_lock = threading.Lock()
yolo_inference_lock = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire Vision | Local Demo</title>
<style>
:root{color-scheme:dark;--bg:#080b12;--panel:#111722;--line:#263244;--muted:#98a7bb;--text:#eef4ff;--orange:#ff7a32;--cyan:#68d9d0}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% 0,#1c2634 0,transparent 42%),var(--bg);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,Segoe UI,sans-serif;min-height:100vh}.wrap{max-width:1050px;margin:0 auto;padding:48px 22px 72px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:62px}.brand{font-weight:800;letter-spacing:.08em;font-size:13px}.tag{color:var(--cyan);border:1px solid #31534f;border-radius:99px;padding:7px 12px;font-size:12px}.hero{max-width:760px;margin-bottom:34px}.eyebrow{color:var(--orange);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:12px}.hero h1{font-size:clamp(38px,7vw,70px);line-height:1.02;letter-spacing:-.055em;margin:14px 0 18px}.hero p{color:var(--muted);font-size:17px;max-width:640px}.card{background:linear-gradient(145deg,#141c28,#0e141e);border:1px solid var(--line);border-radius:22px;padding:25px;box-shadow:0 22px 60px #0005}.form{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.drop{flex:1;min-width:230px;border:1px dashed #40516a;border-radius:14px;padding:18px;color:var(--muted);background:#0a1019}.drop input{display:block;width:100%;margin-top:9px;color:var(--text)}button{border:0;border-radius:12px;padding:14px 22px;background:var(--orange);color:#17100b;font-weight:800;cursor:pointer}button:disabled{opacity:.6;cursor:wait}.hint{font-size:12px;color:var(--muted);margin-top:13px}.error{margin-top:18px;color:#ffaaa3;background:#30191b;border:1px solid #673537;border-radius:12px;padding:12px 15px}.result{margin-top:24px}.result h2{font-size:18px;margin:0 0 14px}.preview{width:100%;max-height:70vh;object-fit:contain;background:#05070a;border-radius:14px;border:1px solid var(--line)}.meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0;color:var(--muted)}.pill{border-radius:99px;padding:6px 11px;background:#202c3b;color:var(--text);font-size:13px}.pill.fire{background:#552415;color:#ffb18c}.pill.safe{background:#143936;color:#9be3d9}.notice{margin-top:28px;color:var(--muted);font-size:13px;border-left:2px solid var(--cyan);padding-left:13px}.foot{color:#718096;font-size:12px;margin-top:26px}
</style>
<style>
.result{margin-top:30px;padding-top:26px;border-top:1px solid var(--line)}
.result-head{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px}
.result-head h2{font-size:22px;margin:0}.decision{display:inline-flex;align-items:center;gap:9px;border-radius:12px;padding:10px 15px;font-weight:800;letter-spacing:.04em;background:#13312c;color:#8de1c7;border:1px solid #28564b}
.decision.fire{background:#431f19;color:#ffb18c;border-color:#733c2d}.decision-dot{width:9px;height:9px;border-radius:50%;background:currentColor;box-shadow:0 0 14px currentColor}
.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0 20px}.model-card,.metric-card{background:#0b111a;border:1px solid var(--line);border-radius:15px;padding:17px}.model-title{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}.model-title strong{font-size:15px}.model-title small,.metric-label{display:block;color:var(--muted);font-size:12px}.score{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums}.meter{height:7px;background:#263244;border-radius:99px;position:relative;margin:13px 0 9px;overflow:visible}.meter-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--cyan),#a8e7c7)}.meter-fill.fire-fill{background:linear-gradient(90deg,#ffb04d,var(--orange))}.meter-threshold{position:absolute;top:-3px;width:2px;height:13px;background:#fff;opacity:.8}.meter-note{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0 20px}.metric-value{display:block;font-size:22px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.2;margin-top:6px}.metric-sub{display:block;color:var(--muted);font-size:11px;margin-top:5px}.section-label{color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin:22px 0 10px}.video-wrap{background:#05070a;border:1px solid var(--line);border-radius:15px;overflow:hidden}.video-wrap video{display:block;width:100%;max-height:70vh}.download{display:inline-flex;margin-top:15px;border-radius:10px;padding:11px 15px;background:#202c3b;color:var(--text);text-decoration:none;font-weight:700}.download:hover{background:#2b3a4d}.file-name{margin-top:8px;color:var(--text);font-size:13px;overflow-wrap:anywhere}.hint{line-height:1.7}.pill{display:inline-flex;align-items:center;gap:6px}
@media(max-width:680px){.model-grid{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.wrap{padding:28px 15px 48px}.card{padding:18px}.top{margin-bottom:42px}}
</style>
</head>
<body><main class="wrap">
<header class="top"><div class="brand">FIRE VISION / LOCAL</div><div class="tag">MobileNetV3 + YOLO · TFLite</div></header>
<section class="hero"><div class="eyebrow">On-device model demo</div><h1>See what the model sees.</h1><p>Upload an image or video. The model scores each frame for fire and shows the result here in your browser.</p></section>
<section class="card"><form class="form" method="post" enctype="multipart/form-data"><label class="drop">Choose an image or video<input id="media" type="file" name="media" accept="image/*,video/*" required><span id="file-name" class="file-name">No file selected</span></label><button id="submit" type="submit">Analyze media</button></form><div class="hint">Images: PNG, JPG, WEBP · Video: MP4, AVI, MOV, MKV, WEBM · Maximum 100 MB. Video is analyzed at up to 10 samples per second.</div>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if image_result %}<section class="result" aria-label="Image analysis results"><div class="result-head"><h2>Image analysis</h2><div class="decision {{ 'fire' if image_result.fire else '' }}"><span class="decision-dot"></span>{{ 'FIRE DETECTED' if image_result.fire else 'NO FIRE CONFIRMED' }}</div></div><p class="section-label">Model scores · confidence and decision thresholds</p><div class="model-grid"><article class="model-card"><div class="model-title"><div><strong>MobileNetV3 · gate</strong><small>Opens YOLO confirmation when score reaches gate</small></div><span class="score">{{ '%.1f'|format(image_result.classifier_probability * 100) }}%</span></div><div class="meter"><div class="meter-fill fire-fill" style="width:{{ [image_result.classifier_probability * 100, 100]|min }}%"></div><span class="meter-threshold" style="left:{{ gate_threshold * 100 }}%"></span></div><div class="meter-note"><span>0%</span><span>Gate {{ '%.0f'|format(gate_threshold * 100) }}%</span><span>100%</span></div></article><article class="model-card"><div class="model-title"><div><strong>YOLOv8n · confirmation</strong><small>Maximum fire confidence · {{ image_result.box_count }} detected box(es)</small></div><span class="score">{{ '%.1f'|format(image_result.probability * 100) }}%</span></div><div class="meter"><div class="meter-fill fire-fill" style="width:{{ [image_result.probability * 100, 100]|min }}%"></div><span class="meter-threshold" style="left:{{ fire_threshold * 100 }}%"></span></div><div class="meter-note"><span>0%</span><span>Fire threshold {{ '%.0f'|format(fire_threshold * 100) }}%</span><span>100%</span></div></article></div><div class="metric-grid"><article class="metric-card"><span class="metric-label">YOLO smoke confidence</span><strong class="metric-value">{{ '%.1f'|format(image_result.smoke_probability * 100) }}%</strong><span class="metric-sub">Smoke threshold {{ '%.0f'|format(smoke_threshold * 100) }}%</span></article><article class="metric-card"><span class="metric-label">Cascade decision</span><strong class="metric-value">{{ 'FIRE' if image_result.fire else 'SAFE' }}</strong><span class="metric-sub">YOLO confirmation, not classifier alone</span></article></div><img class="preview" src="data:image/jpeg;base64,{{ image_result.image }}" alt="Image annotated with model detections"></section>{% endif %}
{% if video_result %}<section class="result" aria-label="Video analysis results"><div class="result-head"><h2>Video analysis</h2><div class="decision {{ 'fire' if video_result.fire_frames else '' }}"><span class="decision-dot"></span>{{ 'FIRE DETECTED' if video_result.fire_frames else 'NO FIRE CONFIRMED' }}</div></div><div class="metric-grid"><article class="metric-card"><span class="metric-label">Analyzed samples</span><strong class="metric-value">{{ video_result.analyzed_frames }}</strong><span class="metric-sub">of up to {{ '%.1f'|format(video_result.duration * 10) }} at 10 samples/sec</span></article><article class="metric-card"><span class="metric-label">Fire alarm samples</span><strong class="metric-value">{{ video_result.fire_frames }}</strong><span class="metric-sub">after 8-of-10 temporal filter</span></article><article class="metric-card"><span class="metric-label">Smoke alarm samples</span><strong class="metric-value">{{ video_result.smoke_frames }}</strong><span class="metric-sub">after 8-of-10 temporal filter</span></article><article class="metric-card"><span class="metric-label">Video duration</span><strong class="metric-value">{{ '%.1f'|format(video_result.duration) }} s</strong><span class="metric-sub">{{ video_result.fps|round|int }} FPS playback</span></article></div><p class="section-label">Model scores · peak confidence and cascade activity</p><div class="model-grid"><article class="model-card"><div class="model-title"><div><strong>MobileNetV3 · gate</strong><small>Samples reaching {{ '%.0f'|format(gate_threshold * 100) }}% gate</small></div><span class="score">{{ '%.1f'|format(video_result.classifier_peak * 100) }}%</span></div><div class="meter"><div class="meter-fill fire-fill" style="width:{{ [video_result.classifier_peak * 100, 100]|min }}%"></div><span class="meter-threshold" style="left:{{ gate_threshold * 100 }}%"></span></div><div class="meter-note"><span>Peak classifier score</span><span>{{ video_result.gate_frames }} gate triggers</span></div></article><article class="model-card"><div class="model-title"><div><strong>YOLOv8n · detector</strong><small>Peak fire confidence · {{ video_result.detections }} boxes across samples</small></div><span class="score">{{ '%.1f'|format(video_result.peak * 100) }}%</span></div><div class="meter"><div class="meter-fill fire-fill" style="width:{{ [video_result.peak * 100, 100]|min }}%"></div><span class="meter-threshold" style="left:{{ fire_threshold * 100 }}%"></span></div><div class="meter-note"><span>Peak smoke {{ '%.1f'|format(video_result.peak_smoke * 100) }}%</span><span>Fire threshold {{ '%.0f'|format(fire_threshold * 100) }}%</span></div></article></div><div class="video-wrap"><video controls playsinline preload="metadata" src="{{ url_for('video_stream', video_id=video_result.id) }}"></video></div><a class="download" href="{{ url_for('video_download', video_id=video_result.id) }}">Download annotated MP4</a></section>{% endif %}
</section><div class="notice">Cascade uses the project's MobileNetV3 FP16 classifier to gate the trained YOLOv8n TFLite detector. YOLO confirms fire, detects smoke, and draws boxes; video results use the project's 8-of-10 temporal filter. Model output is not a fire-safety guarantee.</div><footer class="foot">Runs locally on this machine. Uploaded media and processed video are stored temporarily.</footer>
</main><script>const media=document.getElementById('media');media.addEventListener('change',()=>{document.getElementById('file-name').textContent=media.files[0]?.name||'No file selected'});document.querySelector('form').addEventListener('submit',()=>{const b=document.getElementById('submit');b.disabled=true;b.textContent='Analyzing…'});</script></body></html>"""


def infer_rgb(rgb):
    tensor = np.asarray(Image.fromarray(rgb).resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
    with inference_lock:
        interpreter.set_tensor(input_info["index"], tensor)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_info["index"])[0]
    fire_probability = float(probabilities[fire_index])
    return fire_probability, fire_probability >= GATE_THRESHOLD


def infer_yolo(frame_bgr):
    height, width = frame_bgr.shape[:2]
    scale = min(YOLO_SIZE / width, YOLO_SIZE / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (YOLO_SIZE - resized_width) // 2
    top = (YOLO_SIZE - resized_height) // 2
    canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    with yolo_inference_lock:
        yolo_interpreter.set_tensor(yolo_input_info["index"], tensor)
        yolo_interpreter.invoke()
        predictions = yolo_interpreter.get_tensor(yolo_output_info["index"])[0].T

    boxes = []
    scores = {"fire": 0.0, "smoke": 0.0}
    for class_index, label in enumerate(("smoke", "fire")):
        candidates = predictions[predictions[:, 4 + class_index] >= 0.05]
        if not len(candidates):
            continue
        confidences = candidates[:, 4 + class_index]
        scores[label] = float(confidences.max())
        xyxy = np.empty((len(candidates), 4), dtype=np.float32)
        xyxy[:, 0] = candidates[:, 0] - candidates[:, 2] / 2
        xyxy[:, 1] = candidates[:, 1] - candidates[:, 3] / 2
        xyxy[:, 2] = candidates[:, 0] + candidates[:, 2] / 2
        xyxy[:, 3] = candidates[:, 1] + candidates[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / scale
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
        xywh = xyxy.copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), confidences.tolist(), 0.05, 0.45)
        for index in np.asarray(indices).reshape(-1):
            x1, y1, x2, y2 = xyxy[index].astype(int)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, label, float(confidences[index])))
    return scores["fire"], scores["smoke"], boxes


class CascadeSession:
    def __init__(self):
        self.gate_open_until = -float("inf")
        self.last_yolo_at = -float("inf")
        self.smoke_score = 0.0
        self.fire_history = deque(maxlen=10)
        self.smoke_history = deque(maxlen=10)
        self.last_boxes = []

    def process(self, frame, timestamp):
        classifier_score, _ = infer_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if classifier_score >= GATE_THRESHOLD:
            self.gate_open_until = timestamp + GATE_HOLD_SECONDS
        gate_open = timestamp <= self.gate_open_until
        run_yolo = gate_open or timestamp - self.last_yolo_at >= YOLO_EVERY_SECONDS
        fire_score = 0.0
        boxes = []
        if run_yolo:
            yolo_fire, self.smoke_score, detected_boxes = infer_yolo(frame)
            self.last_yolo_at = timestamp
            fire_score = yolo_fire if gate_open else 0.0
            boxes = detected_boxes if gate_open else [box for box in detected_boxes if box[4] != "fire"]
            self.last_boxes = boxes
        else:
            boxes = self.last_boxes
        self.fire_history.append(fire_score >= FIRE_THRESHOLD)
        self.smoke_history.append(self.smoke_score >= SMOKE_THRESHOLD)
        return {"fire_score": fire_score, "smoke_score": self.smoke_score,
                "fire": sum(self.fire_history) >= 8, "smoke": sum(self.smoke_history) >= 8,
                "boxes": boxes, "classifier_score": classifier_score, "yolo_ran": run_yolo}


def draw_result(frame, result):
    annotated = frame.copy()
    for x1, y1, x2, y2, label, confidence in result["boxes"]:
        color = (50, 95, 255) if label == "fire" else (210, 210, 210)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    label = "FIRE" if result["fire"] else "SMOKE" if result["smoke"] else "NO ALARM"
    color = (50, 95, 255) if result["fire"] else (65, 205, 180)
    text = (f"{label} | YOLO fire {result['fire_score']:.1%} "
            f"smoke {result['smoke_score']:.1%} | MobileNet {result['classifier_score']:.1%}")
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (12, 17, 25), -1)
    cv2.putText(annotated, text[:110], (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return annotated


@app.route("/", methods=["GET", "POST"])
def home():
    image_result = None
    video_result = None
    error = None
    if request.method == "POST":
        upload = request.files.get("media")
        if not upload or not upload.filename:
            error = "Choose an image or video first."
        else:
            filename = secure_filename(upload.filename)
            extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if extension not in ALLOWED:
                error = "Unsupported file type. Choose an image or video from the supported list."
            else:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
                temp_path = Path(temp.name)
                temp.close()
                upload.save(temp_path)
                if extension in {"png", "jpg", "jpeg", "webp"}:
                    try:
                        with Image.open(temp_path) as source:
                            rgb = np.asarray(source.convert("RGB"))
                        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        result = CascadeSession().process(frame, 0.0)
                        result["fire"] = result["fire_score"] >= FIRE_THRESHOLD
                        result["smoke"] = result["smoke_score"] >= SMOKE_THRESHOLD
                        canvas = draw_result(frame, result)
                        ok, encoded = cv2.imencode(".jpg", canvas)
                        if not ok:
                            raise RuntimeError("Could not encode preview image.")
                        import base64
                        image_result = {"fire": result["fire"], "label": "FIRE" if result["fire"] else "NO FIRE",
                                        "probability": result["fire_score"],
                                        "smoke_probability": result["smoke_score"],
                                        "classifier_probability": result["classifier_score"],
                                        "box_count": len(result["boxes"]),
                                        "image": base64.b64encode(encoded).decode("ascii")}
                    except Exception as exc:
                        error = f"Could not process image: {exc}"
                    finally:
                        temp_path.unlink(missing_ok=True)
                else:
                    output_id = uuid.uuid4().hex
                    output_path = Path(tempfile.gettempdir()) / f"fire-vision-{output_id}.mp4"
                    cap = cv2.VideoCapture(str(temp_path))
                    writer = None
                    try:
                        if not cap.isOpened():
                            raise RuntimeError("Could not open uploaded video.")
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width <= 0 or height <= 0:
                            raise RuntimeError("Video has invalid dimensions.")
                        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Could not create processed MP4 video.")
                        frames = 0
                        fire_frames = 0
                        smoke_frames = 0
                        peak = 0.0
                        peak_smoke = 0.0
                        classifier_peak = 0.0
                        gate_frames = 0
                        analyzed_frames = 0
                        detections = 0
                        next_analysis_time = 0.0
                        last_result = None
                        session = CascadeSession()
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            timestamp = frames / fps
                            if last_result is None or timestamp >= next_analysis_time:
                                last_result = session.process(frame, timestamp)
                                next_analysis_time = timestamp + 0.1
                                peak = max(peak, last_result["fire_score"])
                                peak_smoke = max(peak_smoke, last_result["smoke_score"])
                                classifier_peak = max(classifier_peak, last_result["classifier_score"])
                                gate_frames += int(last_result["classifier_score"] >= GATE_THRESHOLD)
                                fire_frames += int(last_result["fire"])
                                smoke_frames += int(last_result["smoke"])
                                detections += len(last_result["boxes"])
                                analyzed_frames += 1
                            writer.write(draw_result(frame, last_result))
                            frames += 1
                        if frames == 0:
                            raise RuntimeError("Video contains no readable frames.")
                        video_result = {"id": output_id, "frames": frames, "fire_frames": fire_frames,
                                        "smoke_frames": smoke_frames, "peak": peak, "peak_smoke": peak_smoke,
                                        "classifier_peak": classifier_peak, "gate_frames": gate_frames,
                                        "analyzed_frames": analyzed_frames, "detections": detections,
                                        "fps": fps, "duration": frames / fps}
                    except Exception as exc:
                        output_path.unlink(missing_ok=True)
                        error = f"Could not process video: {exc}"
                    finally:
                        cap.release()
                        if writer is not None:
                            writer.release()
                        temp_path.unlink(missing_ok=True)
    return render_template_string(PAGE, image_result=image_result, video_result=video_result,
                                  error=error, fire_threshold=FIRE_THRESHOLD, gate_threshold=GATE_THRESHOLD,
                                  smoke_threshold=SMOKE_THRESHOLD)


@app.get("/video/<video_id>")
def video_stream(video_id):
    if len(video_id) != 32 or not all(char in "0123456789abcdef" for char in video_id):
        return "Invalid video id", 400
    path = Path(tempfile.gettempdir()) / f"fire-vision-{video_id}.mp4"
    if not path.is_file():
        return "Processed video not found", 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/video/<video_id>/download")
def video_download(video_id):
    if len(video_id) != 32 or not all(char in "0123456789abcdef" for char in video_id):
        return "Invalid video id", 400
    path = Path(tempfile.gettempdir()) / f"fire-vision-{video_id}.mp4"
    if not path.is_file():
        return "Processed video not found", 404
    return send_file(path, mimetype="video/mp4", as_attachment=True, download_name="fire-vision-annotated.mp4")


@app.errorhandler(413)
def too_large(_error):
    return render_template_string(PAGE, image_result=None, video_result=None,
                                  error="File is larger than 100 MB.", fire_threshold=FIRE_THRESHOLD,
                                  gate_threshold=GATE_THRESHOLD, smoke_threshold=SMOKE_THRESHOLD), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
