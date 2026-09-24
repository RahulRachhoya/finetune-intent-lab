"""Classic ML baseline: TF-IDF (word + char n-grams) -> linear classifier, C tuned on validation."""
import time

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from finetune_intent_lab import data, metrics

MODELS = {
    "logreg": lambda C: LogisticRegression(C=C, max_iter=3000),
    "linear_svm": lambda C: LinearSVC(C=C),
}


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


def main():
    ds = data.load()
    for name, make in MODELS.items():
        best = None
        for C in [0.3, 1, 3, 10, 30]:
            pipe = make_pipeline(features(), make(C)).fit(ds.train.texts, ds.train.labels)
            acc = metrics.score(ds.val.labels, pipe.predict(ds.val.texts).tolist())["accuracy"]
            print(f"{name} C={C}: val acc {acc:.4f}")
            if best is None or acc > best[0]:
                best = (acc, C)

        with metrics.start_run(f"sklearn-{name}"):
            C = best[1]
            mlflow.log_params({"approach": "classic_ml", "model": name, "features": "tfidf word1-2 + char_wb2-5", "C": C})
            t = time.perf_counter()
            pipe = make_pipeline(features(), make(C)).fit(ds.train.texts, ds.train.labels)
            mlflow.log_metric("train_seconds", time.perf_counter() - t)
            for split in ("val", "test"):
                s = getattr(ds, split)
                t = time.perf_counter()
                pred = pipe.predict(s.texts).tolist()
                metrics.log_split(split, s.labels, pred, time.perf_counter() - t)


if __name__ == "__main__":
    main()
