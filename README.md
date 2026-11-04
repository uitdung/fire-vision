# Fire Vision demo

Standalone Flask demo using the project's MobileNetV3 + YOLOv8n TFLite cascade for image and video analysis.

## Included files

- `app.py` — web app and cascade inference
- `models/mobilenetv3s_224_fp16.tflite` — first-stage fire classifier
- `models/yolov8n_416_int8.tflite` — second-stage fire/smoke detector
- `models/*.json` — model metadata and thresholds

## Requirements

- Debian 12 x86_64 VPS
- Python 3.11
- At least 4 GB RAM recommended for TensorFlow Lite and video uploads

## Install and run

```sh
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential libglib2.0-0 libgl1
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
gunicorn --workers 1 --threads 2 --timeout 600 --bind 127.0.0.1:8000 app:app
```

Run the commands from this directory. Gunicorn listens only on localhost; put Nginx or another reverse proxy in front and configure its request body limit to at least `100m` and proxy timeout to at least `600s` for video uploads. Do not expose Flask's built-in development server to the public internet.

For a local smoke test, run `python app.py`; it binds to `127.0.0.1:5000`.

## Create a Git repository

Initialize and push this directory as its own repository. On the VPS, clone the repository, then follow **Install and run** from the cloned directory. The model files are included in the repository, so no separate download is needed.

The demo stores uploaded and processed videos in the system temporary directory. It is intended for testing, not as a production fire-alarm service.
