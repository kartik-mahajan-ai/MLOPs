
# MP088 — Power Loom Output (Nagpur)

**Group:** G262  **Problem:** MP088  **Faculty:** Yadit Kumar  **Batch:** 6

| Enrolment no. | Name |
|---|---|
| S24CSEU0592 | Tanisha Kumar |
| S24CSEU0598 | Muhammad Amaan Khan |
| S24CSEU0642 | Kartik Lamba |
| S24CSEU0640 | Kartik Mahajan |

Predict `metres_per_shift` (regression, MAE) for each loom shift.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install dvc
dvc pull    # or copy train.csv and later.csv from the zip into data/raw/
```

## Run the pipeline (from the repo root, with `src/` on the path)

```bash
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
python -m mp088.data_checks      # stage 1: the seven data problems -> reports/findings.yaml
python -m mp088.train            # stage 2: clean, 4 settings tracked in MLflow, best -> v1 @champion
pytest -q                        # tests (no CSVs needed)
```

`python -m mp088.train` must run on a fresh `mlflow.db` (it registers version 1).

## Layout

| Path | What |
|---|---|
| `params.yaml` | paths, cleaning rules, model settings (read by every module) |
| `src/mp088/data_checks.py` | finds the seven data problems |
| `src/mp088/cleaning.py` | fixes them (train.csv only) |
| `src/mp088/model.py` | scikit-learn pipeline: impute + scale + one-hot, then regressor |
| `src/mp088/train.py` | v1 training with MLflow tracking and registry |
| `app/main.py` | FastAPI: `POST /predict`, `GET /health` |

## Results

- v1: Ridge, validation MAE 2.33 m (best of 4 settings), registered as `mp088-loom-model` v1 `@champion`.

## Docker

```bash
docker build -t mp088-api:1.0 .
docker run -p 8000:8000 mp088-api:1.0
```
