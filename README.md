# Local Keyboard Detection with Roboflow Inference CLI

This workspace is set up to run the Roboflow model `keyboard-key-recognition-kw7nc/14` on-device through a local inference server.

## What "local" means here

Inference runs on your device, not through Roboflow's hosted inference API, so you avoid hosted API latency and per-request inference costs.

There is still one important caveat: the model weights usually need a one-time authenticated download using your Roboflow API key. After the weights are cached locally, inference runs against the local server.

The CLI implementation mounts host `/tmp` into the container. The wrapper in this folder redirects the relevant `/tmp` cache paths into workspace-backed directories such as `.roboflow-cache/`, so the model cache survives normal container restarts and does not live only in a temporary directory.

## Files

- `setup_inference.sh`: creates a Python 3.12 virtual environment and installs `inference-cli`
- `start_inference.sh`: starts the local inference server on port `9001`
- `run_keyboard_detection.sh`: runs your keyboard model on a local image
- `.env.inference.example`: template for your local configuration

## First-time setup

1. Create the runtime:

```bash
./setup_inference.sh
```

2. Create your local env file and add your API key:

```bash
cp .env.inference.example .env.inference
```

3. Edit `.env.inference` and set:

```bash
ROBOFLOW_API_KEY=your_real_key_here
```

## Start the local server

```bash
./start_inference.sh
```

This starts the Roboflow inference server on `http://localhost:9001`.
On the first run, Docker may need to pull the server image and download model weights.

## Run inference on your keyboard image

Use the image already in this folder:

```bash
./run_keyboard_detection.sh
```

Or pass a different image:

```bash
./run_keyboard_detection.sh /path/to/image.jpg
```

Visualized results are written to `outputs/`.
The first successful request also primes the local model cache.

## Export a Project 4 target map

The robot side should consume the normalized target-map JSON, not raw Roboflow output.

After saving three raw detector JSON files from the same calibration pose, run:

```bash
python scripts/export_target_map.py \
  --raw-json outputs/batch_detection/raw/frame0.json outputs/batch_detection/raw/frame1.json outputs/batch_detection/raw/frame2.json \
  --target-text "hello 2026" \
  --image WIN_20260509_15_59_08_Pro.jpg \
  --output outputs/target_map.json \
  --overlay outputs/target_overlay.jpg
```

The target map uses the v1 contract:

- markerless QWERTZ key-grid fitting
- three-frame repeatability gate
- image-space key centers only
- visible end-effector tip detection

To test the final image-space correction without robot hardware:

```bash
python scripts/simulate_visual_servo.py --target-x 500 --target-y 400 --tip-x 200 --tip-y 200
```

## Current machine notes

This machine already has:

- `python3.12`
- `docker`

The CLI could not be installed on Python `3.13`, so the setup uses a dedicated Python `3.12` virtual environment at `.venv312/`.

## Supported deployment targets

Roboflow Inference CLI supports:

- x86 CPU devices
- ARM64 CPU devices
- NVIDIA GPU devices
- NVIDIA Jetson devices

The same setup pattern applies on Raspberry Pi and other ARM CPU devices as long as Docker and a supported Python version are available.

For RTSP cameras, you typically keep the same local inference server and feed RTSP frames or streams into your application against `http://localhost:9001`.

## Important limitation

This setup avoids Roboflow's hosted inference API during prediction, but it is not a fully air-gapped deployment unless you have a way to pre-download/export the model weights and keep them cached locally. Standard Roboflow Inference still expects authenticated access when the model is first fetched.
