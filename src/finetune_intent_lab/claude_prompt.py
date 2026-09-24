"""Prompted-Claude baseline: label list + k examples per label in a cached system prompt.

Runs on Amazon Bedrock through global cross-region inference profiles, using the standard AWS
credential chain (AWS_PROFILE / AWS_REGION). Every run costs money, so --limit evaluates a
random sample of each split instead of all 4,000 queries.

    python -m finetune_intent_lab.claude_prompt --limit 300
"""
import argparse
import random
import time

import os

import anthropic
from anthropic import AnthropicBedrock
import mlflow

from finetune_intent_lab import data, metrics

# Anthropic list prices, $/1M tokens (input, output); Bedrock bills separately, so treat cost as an
# estimate. Cache writes are 1.25x input (5-min TTL), cache reads 0.1x input.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def system_prompt(ds: data.Banking77, shots: int) -> str:
    rng = random.Random(data.SEED)
    by_label: dict[int, list[str]] = {}
    for text, y in zip(ds.train.texts, ds.train.labels):
        by_label.setdefault(y, []).append(text)
    lines = [
        "You classify online-banking customer queries into exactly one intent label.",
        "Reply with the label only, copied exactly from the list below. No other text.",
        "",
        "Labels, each followed by example queries:",
    ]
    for i, name in enumerate(ds.label_names):
        examples = rng.sample(by_label[i], shots) if shots else []
        lines.append(f"- {name}" + "".join(f"\n    e.g. {e}" for e in examples))
    return "\n".join(lines)


def sample(split: data.Split, limit: int | None) -> data.Split:
    if not limit or limit >= len(split.texts):
        return split
    idx = random.Random(data.SEED).sample(range(len(split.texts)), limit)
    return data.Split([split.texts[i] for i in idx], [split.labels[i] for i in idx])


BEDROCK_IDS = {
    "claude-opus-5": "global.anthropic.claude-opus-5",
    "claude-sonnet-5": "global.anthropic.claude-sonnet-5",
    "claude-haiku-4-5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def classify(client, model, system, text, effort):
    kwargs = {}
    if model != "claude-haiku-4-5":  # Haiku 4.5 rejects the effort parameter
        kwargs["output_config"] = {"effort": effort}
    return client.messages.create(
        model=BEDROCK_IDS[model],
        max_tokens=256,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}],
        **kwargs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-opus-5", choices=sorted(PRICES))
    p.add_argument("--shots", type=int, default=1, help="train examples per label in the prompt")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--limit", type=int, default=300, help="queries per split (0 = full split)")
    args = p.parse_args()

    ds = data.load()
    system = system_prompt(ds, args.shots)
    index = {name: i for i, name in enumerate(ds.label_names)}
    client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "ap-south-1"))
    price_in, price_out = PRICES[args.model]

    with metrics.start_run(f"{args.model}-{args.shots}shot"):
        mlflow.log_params({"approach": "claude_prompted", "model": args.model, "shots": args.shots,
                           "effort": args.effort, "limit": args.limit})
        for split in ("val", "test"):
            s = sample(getattr(ds, split), args.limit)
            preds, cost, refusals = [], 0.0, 0
            t = time.perf_counter()
            for n, text in enumerate(s.texts, 1):
                try:
                    r = classify(client, args.model, system, text, args.effort)
                except anthropic.BadRequestError as e:
                    raise SystemExit(f"Bad request (check model/params): {e.message}")
                except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                    raise SystemExit(f"AWS credentials lack Bedrock access for {args.model}: {e.message}")
                u = r.usage
                cost += (u.input_tokens * price_in + (u.cache_creation_input_tokens or 0) * price_in * 1.25
                         + (u.cache_read_input_tokens or 0) * price_in * 0.1 + u.output_tokens * price_out) / 1e6
                if r.stop_reason == "refusal":
                    refusals += 1
                    preds.append(-1)
                    continue
                answer = "".join(b.text for b in r.content if b.type == "text").strip()
                preds.append(index.get(answer, -1))
                print(f"  {split} {n}/{len(s.texts)}  cost ${cost:.3f}", end="\r")
            print()
            metrics.log_split(split, s.labels, preds, time.perf_counter() - t)
            mlflow.log_metrics({f"{split}_usd_per_1k": 1000 * cost / len(s.texts), f"{split}_refusals": refusals})
            print(f"[{split}] ${1000 * cost / len(s.texts):.2f} per 1K queries")


if __name__ == "__main__":
    main()
