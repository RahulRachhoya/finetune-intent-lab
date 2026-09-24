"""README charts, rendered in a light and a dark variant for GitHub's two themes.

    python -m finetune_intent_lab.charts export   # MLflow -> docs/results.json (committed)
    python -m finetune_intent_lab.charts render   # docs/results.json -> docs/charts/*.png

Rendering only needs the committed JSON, so the charts can be rebuilt without mlflow.db.
Style: one accent hue for the fine-tuned model, recessive gray for context, direct labels.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

RESULTS = Path("docs/results.json")
CHART_DIR = Path("docs/charts")

# (run name in MLflow, display label, evaluated on the 300-query sample?)
APPROACHES = [
    ("qwen2.5-1.5b-qlora", "Qwen2.5-1.5B + QLoRA", False),
    ("sklearn-logreg", "TF-IDF + logistic regression", False),
    ("torch-embeddingbag", "PyTorch EmbeddingBag", False),
    ("claude-opus-5-1shot", "Claude Opus 5, prompted", True),
    ("claude-haiku-4-5-1shot", "Claude Haiku 4.5, prompted", True),
    ("qwen2.5-1.5b-zero-shot", "Qwen2.5-1.5B zero-shot", False),
]
HIGHLIGHT = "qwen2.5-1.5b-qlora"

THEMES = {
    "light": {
        "surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "accent": "#2a78d6", "context": "#b5b3aa",
    },
    "dark": {
        "surface": "#0d1117", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "accent": "#3987e5", "context": "#5f5e59",
    },
}


def export():
    import mlflow
    from mlflow import MlflowClient

    from finetune_intent_lab import metrics

    mlflow.set_tracking_uri(metrics.TRACKING_URI)
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[metrics.EXPERIMENT], order_by=["start_time DESC"])
    out = {"approaches": {}, "qlora_loss": {}}
    for name, _, _ in APPROACHES:
        row = runs[runs["tags.mlflow.runName"] == name].iloc[0]  # latest run with that name
        out["approaches"][name] = {
            k: float(row[f"metrics.test_{k}"]) for k in ("accuracy", "macro_f1", "invalid_rate", "ms_per_example")
        }
        if name == HIGHLIGHT:
            for key in ("loss", "eval_loss"):
                out["qlora_loss"][key] = [[m.step, m.value] for m in client.get_metric_history(row.run_id, key)]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS}")


def style(ax, t, grid_axis):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, t, title, subtitle):
    fig.text(0.02, 0.96, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.895, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def hbar(ax, y, width, height, color, radius_px, surface):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.bbox.width
    y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height
    r = radius_px * x_per_px
    ax.add_patch(FancyBboxPatch(
        (0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=y_per_px / x_per_px, facecolor=color, edgecolor=surface, linewidth=0,
    ))
    ax.add_patch(Rectangle((0, y - height / 2), min(width, 2 * r), height, facecolor=color, linewidth=0))


def chart_accuracy(data, t, path):
    rows = sorted(APPROACHES, key=lambda a: data["approaches"][a[0]]["accuracy"])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.30, right=0.95, top=0.78, bottom=0.14)
    style(ax, t, "x")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.canvas.draw()
    bar_h = 20 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height  # 20px bars
    for i, (name, label, sampled) in enumerate(rows):
        acc = 100 * data["approaches"][name]["accuracy"]
        color = t["accent"] if name == HIGHLIGHT else t["context"]
        hbar(ax, i, acc, bar_h, color, 4, t["surface"])
        ax.text(acc + 1.2, i, f"{acc:.1f}%", va="center", color=t["ink"], fontsize=9.5,
                fontweight="bold" if name == HIGHLIGHT else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label + ("*" if sampled else "") for _, label, sampled in rows], fontsize=9.5)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
    titles(fig, t, "Fine-tuning wins: test accuracy on Banking77",
           "77-way intent classification. Higher is better.")
    fig.text(0.02, 0.03, "* Claude scored on a fixed 300-query random sample of the test set; "
             "all others on the full 3,076 queries.", color=t["muted"], fontsize=8)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def chart_tradeoff(data, t, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.2)
    style(ax, t, "both")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 8000)
    ax.set_ylim(30, 100)
    # label offsets (points) chosen per point so labels clear each other and the markers
    offsets = {
        "qwen2.5-1.5b-qlora": (0, 13, "center"), "sklearn-logreg": (-6, 13, "left"),
        "torch-embeddingbag": (-6, -14, "left"), "claude-opus-5-1shot": (0, -15, "center"),
        "claude-haiku-4-5-1shot": (0, 13, "center"), "qwen2.5-1.5b-zero-shot": (10, 0, "left"),
    }
    for name, label, sampled in APPROACHES:
        a = data["approaches"][name]
        x, y = a["ms_per_example"], 100 * a["accuracy"]
        hi = name == HIGHLIGHT
        ax.scatter([x], [y], s=110 if hi else 80, color=t["accent"] if hi else t["context"],
                   edgecolors=t["surface"], linewidths=2, zorder=3)
        dx, dy, ha = offsets[name]
        ax.annotate(f"{label}{'*' if sampled else ''}  {y:.1f}%", (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=t["ink"],
                    fontweight="bold" if hi else "normal")
    ticks = [0.1, 1, 10, 100, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms", "1 s"])
    ax.set_yticks(range(30, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(30, 101, 10)])
    ax.set_xlabel("Latency per query (log scale)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Test accuracy", color=t["ink2"], fontsize=9)
    titles(fig, t, "Accuracy vs latency: top-left is best",
           "QLoRA is the most accurate; classic ML is 2 points behind at ~300x lower latency; prompted Claude trails both.")
    fig.text(0.02, 0.03, "Local models: batched on an RTX 5050 laptop GPU (classic ML on CPU). "
             "Claude: one sequential Bedrock API call per query.  * 300-query sample.",
             color=t["muted"], fontsize=8)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def chart_loss(data, t, path):
    loss = data["qlora_loss"]["loss"]
    ev = data["qlora_loss"]["eval_loss"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.09, right=0.84, top=0.78, bottom=0.14)
    style(ax, t, "y")
    ax.set_yscale("log")
    ax.plot([s for s, _ in loss], [v for _, v in loss], color=t["accent"], linewidth=2, zorder=3)
    ax.scatter([s for s, _ in ev], [v for _, v in ev], s=64, color=t["surface"],
               edgecolors=t["ink2"], linewidths=2, zorder=4)
    s, v = loss[-1]
    ax.annotate(f"train loss  {v:.3f}", (s, v), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=t["ink"])
    s, v = ev[-1]
    ax.annotate(f"validation loss  {v:.3f}", (s, v), xytext=(8, 4), textcoords="offset points",
                va="bottom", fontsize=9, color=t["ink"])
    ax.axvline(ev[0][0], color=t["axis"], linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(ev[0][0], ax.get_ylim()[1], " epoch 1 ends", va="top", fontsize=8.5, color=t["muted"])
    ax.set_xlim(0, loss[-1][0] * 1.02)
    ax.set_xlabel("Training step (batch of 16)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Cross-entropy on label tokens (log)", color=t["ink2"], fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    titles(fig, t, "QLoRA training converges within the first epoch",
           "Training loss (line) and validation loss at each epoch end (circles). "
           "18 min on an RTX 5050, 2.1 GB VRAM.")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def chart_before_after(data, t, path):
    before, after = data["approaches"]["qwen2.5-1.5b-zero-shot"], data["approaches"][HIGHLIGHT]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.06, right=0.97, top=0.72, bottom=0.12, wspace=0.35)
    panels = [("Test accuracy (higher is better)", "accuracy", 100),
              ("Invalid outputs (lower is better)", "invalid_rate", 8)]
    for ax, (label, key, ymax) in zip(axes, panels):
        style(ax, t, "y")
        vals = [100 * before[key], 100 * after[key]]
        ax.bar([0, 1], vals, width=0.42, color=[t["context"], t["accent"]], zorder=3)
        for x, v in zip([0, 1], vals):
            ax.text(x, v + ymax * 0.02, f"{v:.1f}%", ha="center", va="bottom", color=t["ink"],
                    fontsize=10, fontweight="bold" if x == 1 else "normal")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Zero-shot", "After QLoRA"], fontsize=9.5)
        ax.set_ylim(0, ymax * 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
        ax.set_title(label, color=t["ink2"], fontsize=9.5, loc="left")
    titles(fig, t, "Same 1.5B model, before and after fine-tuning",
           "An invalid output is any reply that is not exactly one of the 77 labels (scored as wrong).")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def render():
    data = json.loads(RESULTS.read_text())
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    for name, fn in [("accuracy", chart_accuracy), ("tradeoff", chart_tradeoff),
                     ("training-loss", chart_loss), ("before-after", chart_before_after)]:
        for theme, t in THEMES.items():
            fn(data, t, CHART_DIR / f"{name}-{theme}.png")
    print(f"wrote {len(list(CHART_DIR.glob('*.png')))} charts to {CHART_DIR}")


if __name__ == "__main__":
    {"export": export, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
