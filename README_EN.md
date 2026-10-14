# yolo_Blood

[中文](README.md) | English

An experimental blood-bag image project built with Ultralytics YOLO11, containing two separate tasks:

- **Label detection:** YOLO11-OBB locates label stickers and returns oriented boxes with four corner points. The detector has one class, `blood_label`.
- **Bag-content classification:** YOLO11-cls distinguishes `plasma` from `red_blood_cell`, returning an image-level class and confidence.

The repository also includes PC camera/Web demos, RK3576 RKNN inference, conversion experiments, and troubleshooting notes. Label localization does not perform OCR, and bag-content classification does not determine ABO/Rh blood group. This README provides no evidence of clinical validation or suitability for medical use. Do not use these outputs for diagnosis, compatibility testing, or transfusion decisions.

## Repository map

```text
yolo_Blood/
├── src/blood_label/         # OBB data preparation, training, PC inference and demos
│   ├── crops/              # Label-cropping utility
│   ├── raw/                # Some committed samples/labels, not a full training set
│   └── rknn/               # Standard OBB board inference, Web server and templates
├── src/blood_classify/      # Classification data preparation, training and inference
├── configs/                # Dataset configurations
├── scripts/rknn/           # RKOPT conversion, ONNX inspection and opset utilities
├── runtime/rknn/           # Committed OBB RKNN model, runtime libraries and Lite2 wheel
├── runs/                   # Selected training outputs, weights and ONNX files
├── templates/              # Templates for root-level Web servers
├── server_remote.py        # RKOPT Web server, port 5000
├── server_hd.py            # RKOPT HD camera/capture-and-crop server, port 5001
└── docs/                   # Implementation and troubleshooting notes
```

The working directory matters: run OBB commands from `src/blood_label/` and classification commands from `src/blood_classify/`. Each example below identifies its directory. Replace `/absolute/path/...` with paths on your machine.

## PC environment

Imports in the code identify the core dependencies as Python, PyTorch, Ultralytics, OpenCV and NumPy. Web demos additionally require Flask. There is no repository-wide `requirements.txt`, `pyproject.toml` or dependency lockfile. This is a starting installation example, not a tested, pinned environment:

```bash
# Repository root
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# On Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy flask
```

Choose the appropriate PyTorch installation for your CPU/CUDA environment first, then check installed versions and GPU availability. Training defaults to `--device 0`; use `--device cpu` without a supported CUDA GPU. Camera windows require a working camera, graphical desktop and GUI-enabled OpenCV.

[CLAUDE.md](CLAUDE.md) records the original development environment, but is not a reproducible dependency specification. Ultralytics may download pretrained weights when they are absent. Load only trusted model weights.

## 1. Blood-bag label detection (OBB)

### Prepare data

Place authorized, de-identified `*.origin.jpg` originals and corresponding manually annotated `*.render.jpg` images in `src/blood_label/data/raw/`. The extractor relies on this project's annotation style and geometric rules; it is not a general-purpose annotation converter.

```bash
# Start at the repository root
cd src/blood_label
python extract_labels.py --raw data/raw --output data/raw
python validate_data.py --raw data/raw
python split_dataset.py --raw data/raw --output data/datasets/blood_label --manual data/manual_label --seed 42
```

- Extracted labels use `class x1 y1 x2 y2 x3 y3 x4 y4` with **absolute pixel coordinates**.
- The split script uses approximately 70/20/10 train/val/test splits, normalizes coordinates to 0–1, and creates `images/` and `labels/` under each split.
- Samples with empty labels are copied to a manual-review directory; they are not used as negative training examples.
- Pairing sorts filenames and uses `zip`, rather than matching unique identifiers. Verify pair alignment and counts for new data. Check near-duplicate captures for leakage across training and validation splits.
- Output directories are reused without removing old files. Use a fresh output directory for a different dataset version.

Edit [configs/blood_label.yaml](configs/blood_label.yaml) so `path` is the **absolute path** of the generated dataset, avoiding differences in Ultralytics dataset-directory settings and relative-path resolution. Retain `train: train/images`, `val: val/images`, `test: test/images`, and class `0: blood_label`.

The complete default `data/datasets/` training directories are not committed. Sample counts in historical notes do not guarantee the contents of a fresh checkout.

### Train

```bash
# Working directory: src/blood_label
python train.py --data ../../configs/blood_label.yaml --weights yolo11n-obb.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

The script defaults to `--project ../../runs/obb --name blood_label` and enables rotation/flip augmentation. Use the actual `best.pt` path printed after training; repeated runs may increment the output directory. Replace `--device 0` with `--device cpu` for CPU training.

### Image, directory, camera and Web inference

These examples explicitly select the committed n-2 OBB weights. Replace `--weights` with your own training output when appropriate:

```bash
# Working directory: src/blood_label
python detect.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --source /absolute/path/image.jpg --conf 0.25
python detect.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --source /absolute/path/images --save-txt
python demo.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --camera 0
python server_pc.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --cam 0 --port 5000
```

`detect.py` defaults to `../../runs/detect/blood_label` and writes annotated `det_*` images. `--save-txt` also requests coordinate output from Ultralytics. The custom image writer and Ultralytics may use different incremented output directories, so inspect the logs. Press Q to close the camera demo. The PC Web page is at `http://localhost:5000`.

The camera argument is `--camera` in `demo.py`, but `--cam` in `server_pc.py`. Default weight paths differ among training, image inference and demos; specifying weights explicitly avoids ambiguity.

## 2. Blood-bag content classification

Prepare this structure. Class directory names determine the classes; current inference scripts hard-code `0 = plasma` and `1 = red_blood_cell`:

```text
src/blood_classify/data/datasets/blood_type/
├── train/{plasma,red_blood_cell}/
├── val/{plasma,red_blood_cell}/
└── test/{plasma,red_blood_cell}/
```

`prepare_dataset.py` uses source directories hard-coded in its source. Check those paths before using it; it is not a CLI that accepts arbitrary input directories. You can also organize your own data as shown above. Classification training takes a **directory, not a YAML file**, through `--data`.

```bash
# Start at the repository root
cd src/blood_classify
python train.py --data data/datasets/blood_type --weights yolo11n-cls.pt --epochs 100 --imgsz 224 --batch 32 --device 0
python detect.py --weights /absolute/path/classification/best.pt --source /absolute/path/image.jpg
```

Training defaults to `../../runs/classify/blood_type`. If Windows multiprocessing causes data-loader problems, try `--workers 0`. Image inference prints the predicted class and confidence. Classification `best.pt` and RKNN weights are not committed; train or supply the appropriate models first. If you change classes, verify the hard-coded mappings and exported output order.

## 3. RK3576 deployment

### Environment and model contracts

- Use RKNN-Toolkit2 on the conversion host. The board needs RKNN-Toolkit-Lite2, OpenCV and NumPy; Web servers also need Flask.
- [runtime/rknn](runtime/rknn/) contains an OBB model and runtime files. The committed Lite2 wheel targets `cp310` and `aarch64`; use it only with the matching Python ABI and architecture, not a PC Python 3.12 environment.
- Check compatibility among the model target, Toolkit, Lite2, `librknnrt.so` and board driver. Comparing version numbers across different components is not a compatibility test. Do not blindly overwrite system runtime libraries.
- These scripts do not have a `--model` option. They hard-code the paths below. Place a model with the **correct task and output format** there, or adjust constants in your own deployment copy.

| Entry point | Expected model path | Task/format |
| --- | --- | --- |
| `src/blood_label/rknn/infer.py`, `server.py` | `/root/blood_label/best_rk3576.rknn` | Standard OBB |
| `src/blood_classify/rknn/infer.py` | `/root/blood_classify/best_rk3576.rknn` | Two-class classification |
| `server_remote.py`, `server_hd.py` | `/root/blood_label/best_rkopt_rk3576.rknn` | Four-branch RKOPT OBB |

### Standard model inference

On an RK3576 with its runtime, models and camera configured, run these commands from the repository root:

```bash
python3 src/blood_label/rknn/infer.py --src /absolute/path/image.jpg --conf 0.25
python3 src/blood_label/rknn/infer.py --cam 0
python3 src/blood_label/rknn/server.py

# Requires a separately prepared classification RKNN model
python3 src/blood_classify/rknn/infer.py --src /absolute/path/image.jpg
python3 src/blood_classify/rknn/infer.py --cam 0
```

The OBB Web server uses port 5000. Single-image board scripts save annotated results; camera windows require a graphical environment. Classification preprocessing uses 224×224 RGB input and must match the exported model's normalization and output contract. `infer_ctypes.py` and `infer_min.py` are lower-level C API/minimal troubleshooting tools, not universal replacements for every board setup.

### RKOPT INT8 conversion and Web servers

Read the [RKOPT quantization guide](docs/rkopt-int8-guide.md) first. This route uses a Rockchip-adapted exporter and moves some detection-head operations into CPU postprocessing. An arbitrary standard ONNX or standard OBB RKNN model is not interchangeable with an RKOPT model.

The current [convert_rkopt.py](scripts/rknn/convert_rkopt.py) main program hard-codes local WSL paths and **does not parse the positional arguments shown in the older guide**. In an environment with Toolkit2 installed, call its function from the repository root with your own paths:

```bash
python - <<'PY'
from pathlib import Path
from scripts.rknn.convert_rkopt import convert
out = Path('/absolute/path/rknn_output')
out.mkdir(parents=True, exist_ok=True)
convert('/absolute/path/rkopt.onnx', '/absolute/path/calibration', str(out), target='rk3576')
PY
```

The calibration directory needs at least five `.jpg`/`.png` images and a `calib.txt` containing valid image paths. This is a script precondition, not a guarantee of quantization quality. The function always enables INT8 quantization and writes `best_rkopt_rk3576.rknn`; this artifact is not currently committed.

After placing the RKOPT model on the board, run either server from the repository root and retain the adjacent `templates/` directory:

```bash
python3 server_remote.py  # Port 5000
# Or
python3 server_hd.py      # Port 5001, HD capture and label cropping
```

Their postprocessing is fixed to 640-pixel input, one class, three detection heads and one angle branch: `[1,65,80,80]`, `[1,65,40,40]`, `[1,65,20,20]`, and `[1,1,8400]`. Recheck the contract whenever changing exporter, input size or class count.

## Limitations and troubleshooting

- **Web security:** Flask demos bind to `0.0.0.0` and do not include a complete authentication/TLS deployment setup. Use a trusted, isolated network; do not expose them directly to the public internet. Camera frames, captures and cropped labels may contain personal information. De-identify data and control storage, access and sharing.
- **Versions and historical notes:** The [initialization -7 investigation](docs/rk3576-rknn-init-7-排查.md) and the newer RKOPT guide reach different conclusions about the same driver version. They are investigation records, not a compatibility matrix. Use actual board logs and vendor-matched component versions.
- **Experiments:** Scripts such as `convert_int8.py` and `train_qat.py` preserve experimental approaches. Their names do not establish valid QAT, correct quantization or a speedup.
- **Verification:** This README was prepared from code and configuration inspection. Training, inference and RK3576 execution were not rerun. Historical accuracy/FPS figures are not reproducibility guarantees; validate accuracy, robustness, speed and dataset splits for your deployment.

Further reading: [Development notes](CLAUDE.md) · [RKNN runtime files](runtime/rknn/README.md) · [RKOPT guide](docs/rkopt-int8-guide.md)
