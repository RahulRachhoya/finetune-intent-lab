"""Banking77 loading with a fixed, stratified validation split carved out of train.

The official test split is only used for final numbers; all tuning happens on validation.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dataclasses import dataclass

from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42


@dataclass
class Split:
    texts: list[str]
    labels: list[int]


@dataclass
class Banking77:
    train: Split
    val: Split
    test: Split
    label_names: list[str]


def load(val_fraction: float = 0.1) -> Banking77:
    ds = load_dataset("mteb/banking77")
    train, test = ds["train"], ds["test"]

    names = {}
    for row in train:
        names[row["label"]] = row["label_text"]
    label_names = [names[i] for i in range(len(names))]

    tr_x, va_x, tr_y, va_y = train_test_split(
        train["text"], train["label"], test_size=val_fraction,
        stratify=train["label"], random_state=SEED,
    )
    return Banking77(
        train=Split(tr_x, tr_y),
        val=Split(va_x, va_y),
        test=Split(test["text"], test["label"]),
        label_names=label_names,
    )
