"""Stage 2: train v1 on cleaned train.csv, track every setting in MLflow,
register the best one as version 1 with the alias `champion`.

Run once from the repo root:   python -m mp088.train
"""
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from mp088.cleaning import FEATURES, clean_train, sort_by_date
from mp088.config import ROOT, file_sha256, load_csv, load_params
from mp088.model import build_pipeline, mae


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def main():
    cfg = load_params()
    setup_mlflow(cfg)
    target = cfg["columns"]["target"]
    seed = cfg["train"]["random_state"]
    sha = file_sha256("train", cfg)

    train = sort_by_date(clean_train(load_csv("train", cfg), cfg))
    cut = int(len(train) * (1 - cfg["train"]["val_fraction"]))
    fit_part, val_part = train.iloc[:cut], train.iloc[cut:]

    results = []
    for setting in cfg["train"]["candidates"]:
        with mlflow.start_run(run_name=f"v1-{setting['name']}") as run:
            mlflow.log_param("data_sha256", sha)
            mlflow.log_params(setting)
            mlflow.log_param("train_rows", len(train))

            # compare settings on the last 20% of train (by date) ...
            model = build_pipeline(setting, seed).fit(fit_part[FEATURES], fit_part[target])
            mae_val = mae(model, val_part)
            mlflow.log_metric("mae_val", mae_val)

            # ... then refit on all cleaned train: this is the model we would ship
            model = build_pipeline(setting, seed).fit(train[FEATURES], train[target])
            mlflow.sklearn.log_model(model, name="model",
                                     serialization_format="cloudpickle",
                                     input_example=train[FEATURES].head(3))
            results.append((mae_val, run.info.run_id, model))
            print(f"{setting} mae_val={mae_val:.3f} run={run.info.run_id}")

    best_mae, best_run, best_model = min(results, key=lambda r: r[0])
    name = cfg["mlflow"]["registered_model_name"]
    version = mlflow.register_model(f"runs:/{best_run}/model", name).version
    MlflowClient().set_registered_model_alias(name, "champion", version)

    out = ROOT / cfg["paths"]["model"]
    Path(out).parent.mkdir(exist_ok=True)
    joblib.dump(best_model, out)
    print(f"best run {best_run} mae_val={best_mae:.3f} -> {name} v{version} @champion")


if __name__ == "__main__":
    main()
