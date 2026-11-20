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
dvc pull    # or copy the CSVs from the zip into data/raw/
```

## Data checks

```bash
PYTHONPATH=src python -m mp088.data_checks
```

Prints the result of all seven data-problem checks; results are recorded in `reports/findings.yaml`.
