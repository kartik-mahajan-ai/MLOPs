"""Stage 4: drift (PSI), MAE per shed, retrain v2, promotion rule, holdout predictions.

Run after mp088.train:   python -m mp088.monitor
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mp088.cleaning import FEATURES, clean_train, prepare, sort_by_date
from mp088.config import ROOT, file_sha256, load_csv, load_params
from mp088.model import build_pipeline, mae


def psi(expected, actual, bins=10, eps=1e-4):
    """Population Stability Index. Bin edges = deciles of the reference (train)."""
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def mae_by_group(model, df, group):
    return {g: round(mae(model, part), 4) for g, part in df.groupby(group)}


def main():
    import mlflow
    from mlflow.tracking import MlflowClient
    from mp088.train import setup_mlflow

    cfg = load_params()
    setup_mlflow(cfg)
    client = MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    target, group = cfg["columns"]["target"], cfg["columns"]["group"]
    m = cfg["monitoring"]

    train = sort_by_date(clean_train(load_csv("train", cfg), cfg))
    later = sort_by_date(prepare(load_csv("later", cfg), cfg))
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")

    # 1. drift per numeric column
    psi_scores = {c: round(psi(train[c], later[c], m["psi_bins"]), 4)
                  for c in cfg["columns"]["numeric"]}
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 2. v1 error per shed on later.csv
    by_group = mae_by_group(v1, later, group)
    worst = max(by_group, key=by_group.get)

    # 3. v2 = train + first 70% of later; compare on the last 30%
    cut = int(len(later) * m["retrain_fraction"])
    later_fit, check = later.iloc[:cut], later.iloc[cut:]
    v1_setting = _setting_of_version(client, name, "1")  # v2 = same model type, more data
    v2_data = pd.concat([train, later_fit], ignore_index=True)

    with mlflow.start_run(run_name="v2-retrain") as run:
        mlflow.log_param("data_sha256", file_sha256("train", cfg))
        mlflow.log_params(v1_setting)
        mlflow.log_param("train_rows", len(v2_data))
        v2 = build_pipeline(v1_setting, cfg["train"]["random_state"]).fit(
            v2_data[FEATURES], v2_data[target])
        mae_v1, mae_v2 = mae(v1, check), mae(v2, check)
        mlflow.log_metrics({"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
                            "mae_later_v1": mae(v1, later)})
        for c, v in psi_scores.items():
            mlflow.log_metric(f"psi_{c}", v)
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle",
                                        input_example=v2_data[FEATURES].head(3))
    v2_version = mlflow.register_model(info.model_uri, name).version

    # 4. the client's rule: promote only if v2 is at least 10% better
    promote = mae_v2 <= (1 - m["promote_if_better_by"]) * mae_v1
    champion_version = v2_version if promote else "1"
    client.set_registered_model_alias(name, "champion", champion_version)
    champion = v2 if promote else v1
    joblib.dump(champion, ROOT / cfg["paths"]["model"])

    # 5. final prediction with the champion
    holdout = load_csv("holdout", cfg)
    out = ROOT / cfg["paths"]["predictions"]
    Path(out).parent.mkdir(exist_ok=True)
    pd.DataFrame({"record_id": holdout["record_id"],
                  "prediction": champion.predict(holdout[FEATURES]).round(2)}
                 ).to_csv(out, index=False)

    report = {"psi": psi_scores, "most_drifted_feature": most_drifted,
              "mae_by_group": by_group, "worst_served_group": worst,
              "check_rows": len(check), "mae_v1_on_check": round(mae_v1, 4),
              "mae_v2_on_check": round(mae_v2, 4),
              "improvement": round(1 - mae_v2 / mae_v1, 4),
              "decision": "promote" if promote else "keep",
              "v2_run_id": run.info.run_id, "v2_version": int(v2_version),
              "champion_version": int(champion_version)}
    (ROOT / cfg["paths"]["monitoring_report"]).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _setting_of_version(client, name, version):
    """Read v1's model settings back from the params of the run that made it."""
    run_id = client.get_model_version(name, version).run_id
    params = client.get_run(run_id).data.params
    cfg = load_params()
    for s in cfg["train"]["candidates"]:
        if all(str(v) == params.get(k) for k, v in s.items()):
            return dict(s)
    raise RuntimeError("could not find v1's setting in params.yaml")


if __name__ == "__main__":
    main()
