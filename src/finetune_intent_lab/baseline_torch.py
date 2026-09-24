"""PyTorch baseline: fastText-style EmbeddingBag over word unigrams + hashed bigrams.

Plain training loop (no Trainer) with early stopping on validation accuracy.
"""
import re
import time
import zlib

import mlflow
import torch
from torch import nn

from finetune_intent_lab import data, metrics

EMB_DIM = 128
BIGRAM_BUCKETS = 50_000
EPOCHS = 40
PATIENCE = 5
LR = 3e-3
BATCH = 64


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9]", text.lower())


class Encoder:
    def __init__(self, texts: list[str]):
        vocab = sorted({t for x in texts for t in tokens(x)})
        self.index = {t: i + 1 for i, t in enumerate(vocab)}  # 0 = unknown
        self.size = len(self.index) + 1 + BIGRAM_BUCKETS

    def ids(self, text: str) -> list[int]:
        toks = tokens(text)
        uni = [self.index.get(t, 0) for t in toks]
        # crc32 rather than hash(): str hashes are salted per process.
        bi = [len(self.index) + 1 + (zlib.crc32((a + " " + b).encode()) % BIGRAM_BUCKETS) for a, b in zip(toks, toks[1:])]
        return uni + bi


class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, EMB_DIM, mode="mean")
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(EMB_DIM, n_classes)

    def forward(self, flat_ids, offsets):
        return self.out(self.drop(self.emb(flat_ids, offsets)))


def batches(enc: Encoder, split: data.Split, device, shuffle: bool):
    order = torch.randperm(len(split.texts)).tolist() if shuffle else range(len(split.texts))
    order = list(order)
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        seqs = [enc.ids(split.texts[j]) or [0] for j in idx]
        offsets = torch.tensor([0] + [len(s) for s in seqs[:-1]]).cumsum(0)
        flat = torch.tensor([t for s in seqs for t in s])
        y = torch.tensor([split.labels[j] for j in idx])
        yield flat.to(device), offsets.to(device), y.to(device)


@torch.no_grad()
def predict(model, enc, split, device) -> list[int]:
    model.eval()
    return [p for flat, off, _ in batches(enc, split, device, False) for p in model(flat, off).argmax(1).tolist()]


def main():
    torch.manual_seed(data.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = data.load()
    enc = Encoder(ds.train.texts)
    model = BagClassifier(enc.size, len(ds.label_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    with metrics.start_run("torch-embeddingbag"):
        mlflow.log_params({"approach": "pytorch", "model": "EmbeddingBag uni+bigram", "emb_dim": EMB_DIM,
                           "lr": LR, "batch": BATCH, "max_epochs": EPOCHS, "patience": PATIENCE,
                           "params": sum(p.numel() for p in model.parameters())})
        best_acc, best_state, bad = 0.0, None, 0
        t0 = time.perf_counter()
        for epoch in range(EPOCHS):
            model.train()
            total = 0.0
            for flat, off, y in batches(enc, ds.train, device, True):
                opt.zero_grad()
                loss = loss_fn(model(flat, off), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            val_acc = metrics.score(ds.val.labels, predict(model, enc, ds.val, device))["accuracy"]
            mlflow.log_metrics({"train_loss": total / len(ds.train.texts), "epoch_val_accuracy": val_acc}, step=epoch)
            print(f"epoch {epoch:2d} loss {total / len(ds.train.texts):.4f} val acc {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc, bad = val_acc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        mlflow.log_metrics({"train_seconds": time.perf_counter() - t0, "best_epoch": epoch - bad})
        model.load_state_dict(best_state)
        for split in ("val", "test"):
            s = getattr(ds, split)
            t = time.perf_counter()
            pred = predict(model, enc, s, device)
            metrics.log_split(split, s.labels, pred, time.perf_counter() - t)


if __name__ == "__main__":
    main()
