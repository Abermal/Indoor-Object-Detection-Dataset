# Indoor Object Detection Dataset Training

This project trains a recent object detection model on the Indoor Object Detection Dataset and reports validation mAP for each of the seven classes and for the full validation set. The solution is centered on a Jupyter notebook that can be executed cell-by-cell locally or in Google Colab.

## Problem

Train an object detector on the Indoor Object Detection Dataset with:

- an 80/10/10 train/validation/test split,
- all seven classes represented in each split,
- validation metrics for every class and for the full validation set,
- visualizations of strong and weak validation predictions,
- a data download step when the dataset is not already present.

The dataset is distributed on Zenodo as `Indoor Object Detection Dataset.zip`: https://zenodo.org/records/2654485. Zenodo describes it as 2,213 fully labeled indoor image frames with seven object classes. The local unpacked copy in this repository uses dlib XML annotations under `Indoor Object Detection Dataset/annotation` and image folders `sequence_1` through `sequence_6`.

Local dataset summary:

| Item | Count |
| --- | ---: |
| Images | 2,213 |
| Boxes | 4,595 |
| Classes | 7 |

Class instance counts:

| Class | Boxes |
| --- | ---: |
| chair | 1,662 |
| clock | 280 |
| exit | 545 |
| fireextinguisher | 1,684 |
| printer | 81 |
| screen | 115 |
| trashbin | 228 |

## Public Implementation Check

I searched for existing MMDetection implementations and reusable loaders using queries around `Indoor Object Detection Dataset`, `TUT Indoor Object Detection Dataset`, `mmdetection`, `mmdet`, `github`, and annotation filenames such as `annotation_s1.xml`. I did not find a maintained MMDetection dataset config or a reusable open-source data loader for this exact dataset. The notebook therefore uses a small local loader/converter in `src/indoor_object_detection/dataset.py`.

## Approach

The notebook uses Ultralytics YOLO, a practical recent detector family with a compact API, pretrained weights, built-in mAP reporting, and straightforward Colab installation. The helper package parses dlib XML annotations, builds a deterministic 80/10/10 split while enforcing every class in each split, converts data to YOLO format, and writes `data.yaml`.

Because the images are consecutive video frames, a naive image-level random split would leak near-duplicate frames across train and validation. The splitter therefore keeps each 5-frame group together while searching for the requested split sizes and class coverage.

The package-first approach is feasible for this assignment. It keeps reusable parsing/splitting/conversion logic out of the notebook while still letting a reviewer run everything by executing notebook cells. The setup cell clones this repository automatically when it is opened directly in Colab, then installs the project in editable mode (`pip install -e .`).

## Environment Notes

`uv` is useful locally for reproducible environment management:

```bash
uv sync
uv run jupyter lab
```

Run tests and a tracked command-line training job with:

```bash
uv run pytest -v
uv run train-indoor-detector --epochs 30 --device 0
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Source functions and data models are fully annotated, public APIs include
docstrings, and the package publishes a `py.typed` marker. Enforce both contracts
before committing:

```bash
uvx mypy
uvx ruff check .
```

For a CPU smoke run, reduce the workload with `--epochs 1 --image-size 320
--batch-size 8 --device cpu`. Training metrics and artifacts are written both to
`runs/detect/indoor_yolo_mlflow/` and the local MLflow store.

For Colab, relying on `uv` is unnecessary friction because Colab users expect notebook-local `pip` installation. The notebook therefore runs `pip install -e .` from its setup cell and does not require `uv`.

## Main Files

- `src/indoor_object_detection/`: installable Python package using the standard `src` layout.
- `indoor_object_detection_yolo.ipynb`: end-to-end notebook for download, conversion, training, validation metrics, and visual examples.
- `src/indoor_object_detection/dataset.py`: dataset download, dlib XML parsing, temporal-block split creation, split summary, and YOLO export utilities.
- `src/indoor_object_detection/train.py`: command-line training using Ultralytics' native MLflow parameter, validation-metric, and artifact logging.
- `pyproject.toml`: package metadata and local dependencies.

## Expected Outputs

Running the notebook creates:

- `data/indoor_yolo/`: YOLO-format images, labels, and `data.yaml`.
- `runs/detect/indoor_yolo_train/`: training artifacts.
- `runs/detect/indoor_yolo_val/`: validation metrics, plots, and predictions.
- notebook tables for overall mAP and per-class metrics.
- notebook visualizations of good and bad validation examples.
