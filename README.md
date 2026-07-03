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

Per-sequence class distribution (counts are annotated bounding boxes):

| Sequence | Images | Dataset share | chair | clock | exit | fireextinguisher | printer | screen | trashbin | Total boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequence_1 | 148 | 6.69% | 82 | 31 | 67 | 57 | 0 | 0 | 4 | 241 |
| sequence_2 | 217 | 9.81% | 127 | 25 | 59 | 53 | 23 | 8 | 51 | 346 |
| sequence_3 | 1,154 | 52.15% | 1,302 | 80 | 156 | 965 | 57 | 11 | 114 | 2,685 |
| sequence_4 | 278 | 12.56% | 66 | 41 | 52 | 183 | 0 | 52 | 44 | 438 |
| sequence_5 | 229 | 10.35% | 26 | 16 | 76 | 309 | 0 | 32 | 7 | 466 |
| sequence_6 | 187 | 8.45% | 59 | 87 | 135 | 117 | 1 | 12 | 8 | 419 |
| **Full dataset** | **2,213** | **100.00%** | **1,662** | **280** | **545** | **1,684** | **81** | **115** | **228** | **4,595** |

Normalized per-sequence class distribution (each class is a percentage of all
bounding boxes in that sequence). Percentages use largest-remainder rounding so
each row totals exactly 100.00%:

| Sequence | Images | Dataset share | chair | clock | exit | fireextinguisher | printer | screen | trashbin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequence_1 | 148 | 6.69% | 34.03% | 12.86% | 27.80% | 23.65% | 0.00% | 0.00% | 1.66% |
| sequence_2 | 217 | 9.81% | 36.70% | 7.23% | 17.05% | 15.32% | 6.65% | 2.31% | 14.74% |
| sequence_3 | 1,154 | 52.15% | 48.49% | 2.98% | 5.81% | 35.94% | 2.12% | 0.41% | 4.25% |
| sequence_4 | 278 | 12.56% | 15.07% | 9.36% | 11.87% | 41.78% | 0.00% | 11.87% | 10.05% |
| sequence_5 | 229 | 10.35% | 5.58% | 3.43% | 16.31% | 66.31% | 0.00% | 6.87% | 1.50% |
| sequence_6 | 187 | 8.45% | 14.08% | 20.76% | 32.22% | 27.92% | 0.24% | 2.87% | 1.91% |
| **Full dataset** | **2,213** | **100.00%** | **36.17%** | **6.10%** | **11.86%** | **36.65%** | **1.76%** | **2.50%** | **4.96%** |

Distribution of each sequence across the exported YOLO splits (parentheses
show the percentage of that sequence assigned to the split):

| Sequence | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| sequence_1 | 128 (86.49%) | 10 (6.76%) | 10 (6.76%) | 148 |
| sequence_2 | 172 (79.26%) | 25 (11.52%) | 20 (9.22%) | 217 |
| sequence_3 | 935 (81.02%) | 94 (8.15%) | 125 (10.83%) | 1,154 |
| sequence_4 | 216 (77.70%) | 42 (15.11%) | 20 (7.19%) | 278 |
| sequence_5 | 179 (78.17%) | 30 (13.10%) | 20 (8.73%) | 229 |
| sequence_6 | 140 (74.87%) | 20 (10.70%) | 27 (14.44%) | 187 |
| **All sequences** | **1,770 (79.98%)** | **221 (9.99%)** | **222 (10.03%)** | **2,213** |

Normalized class distribution within each exported YOLO split (each class is a
percentage of all bounding boxes in that split):

| Split | Boxes | chair | clock | exit | fireextinguisher | printer | screen | trashbin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 3,716 | 36.09% | 6.08% | 11.84% | 36.73% | 1.75% | 2.53% | 4.98% |
| Validation | 449 | 37.19% | 5.79% | 11.81% | 36.30% | 1.78% | 2.23% | 4.90% |
| Test | 430 | 35.82% | 6.51% | 12.09% | 36.28% | 1.86% | 2.56% | 4.88% |
| **Full dataset** | **4,595** | **36.17%** | **6.10%** | **11.86%** | **36.65%** | **1.76%** | **2.50%** | **4.96%** |

## Approach

The notebook uses Ultralytics YOLO, a practical recent detector family with a compact API, pretrained weights, built-in mAP reporting, and straightforward Colab installation. The helper package parses dlib XML annotations, builds a deterministic 80/10/10 split while enforcing every class in each split, converts data to YOLO format, and writes `data.yaml`.

Because the images are consecutive video frames, a naive image-level random split would leak near-duplicate frames across train and validation. The splitter therefore keeps each 5-frame group together while searching for the requested split sizes and class coverage.

The package-first approach is feasible for this assignment. It keeps reusable parsing/splitting/conversion logic out of the notebook while still letting a reviewer run everything by executing notebook cells. The setup cell clones this repository automatically when it is opened directly in Colab, then installs the project in editable mode (`pip install -e .`). The repository must be public: Colab's GitHub integration can open a notebook from a private repository, but it does not pass those GitHub credentials to `git clone` inside the runtime.

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

After training, evaluate the selected checkpoint on the held-out test split:

```powershell
uv run evaluate-indoor-detector mlruns/1/<run-id>/artifacts/weights/best.pt --device 0
```

The command writes aggregate and per-class metrics, per-image precision/recall/IoU
scores, best/worst annotated examples, and Ultralytics confusion-matrix plots to
`runs/detect/indoor_yolo_test/`. The checkpoint argument also accepts a `file:` URI.

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
