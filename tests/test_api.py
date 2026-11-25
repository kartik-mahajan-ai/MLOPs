import json
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD = {"loom_age_yr": 20, "thread_count": 120, "power_cuts_min": 30,
        "weaver_exp_yr": 10, "humidity_pct": 60, "shift_hours": 9.0,
        "fabric": "cotton", "shed": "Shed 2"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_number():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert isinstance(r.json()["metres_per_shift"], float)


def test_rule_loom_age_above_55_rejected():
    assert client.post("/predict", json={**GOOD, "loom_age_yr": 56}).status_code == 422
    assert client.post("/predict", json={**GOOD, "loom_age_yr": 55}).status_code == 200


def test_rule_negative_power_cuts_rejected():
    assert client.post("/predict", json={**GOOD, "power_cuts_min": -1}).status_code == 422
    assert client.post("/predict", json={**GOOD, "power_cuts_min": 0}).status_code == 200


def test_missing_field_rejected():
    body = dict(GOOD)
    body.pop("shed")
    assert client.post("/predict", json=body).status_code == 422


def test_json_logging_on_predict(capsys):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip().startswith("{")]
    assert any(entry.get("event") == "prediction" and "prediction" in entry and "latency_ms" in entry for entry in log_lines)
