"""Shared scoring + MLflow logging so every approach is compared the same way."""
import mlflow
from sklearn.metrics import accuracy_score, f1_score

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "banking77-intent"


def score(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    # y_pred may contain -1 for unparseable LLM outputs; they count as wrong.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true)), zero_division=0),
        "invalid_rate": sum(p == -1 for p in y_pred) / len(y_pred),
    }


def start_run(name: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=name)


def log_split(split: str, y_true, y_pred, seconds: float) -> dict[str, float]:
    m = score(y_true, y_pred)
    m["ms_per_example"] = 1000 * seconds / len(y_true)
    mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
    print(f"[{split}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return m
