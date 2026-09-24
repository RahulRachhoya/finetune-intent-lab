# finetune-intent-lab

When is it worth fine-tuning a small open-weight model instead of prompting a frontier LLM?
This repo answers that for one concrete task, intent classification on
[Banking77](https://huggingface.co/datasets/mteb/banking77) (77 intents, 10k train / 3k test
customer queries), by running five approaches through the same data split and metrics and
tracking every run in MLflow.

## Results

Full official test set (3,076 queries), except Claude, which was scored on a fixed random
300-query subset to limit API spend. The last column rescores every model on that same subset.

| Approach | Test acc | Macro-F1 | Latency / query | Cost / 1K queries | Acc on Claude's 300 |
|---|---|---|---|---|---|
| TF-IDF + logistic regression (scikit-learn) | 91.1% | 0.911 | 0.09 ms (CPU) | ~0 | 90.0% |
| fastText-style EmbeddingBag (PyTorch, own training loop) | 90.0% | 0.900 | 0.15 ms | ~0 | - |
| Qwen2.5-1.5B-Instruct, zero-shot (4-bit) | 36.9% | 0.341 | 94 ms | ~0 (local GPU) | - |
| **Qwen2.5-1.5B-Instruct + QLoRA** | **93.4%** | **0.935** | 27 ms (batched) | ~0 (local GPU) | **91.0%** |
| Claude Opus 5, 1-shot-per-label prompt (Bedrock) | - | - | 1,868 ms | $1.70 | 83.7% |
| Claude Haiku 4.5, same prompt (Bedrock) | - | - | 989 ms | $2.10 | 83.7% |

QLoRA training: 18.5M trainable parameters (~1.2% of the model), 2 epochs, 18 minutes and
2.1 GB peak VRAM on a laptop RTX 5050 (8 GB). The trained adapter is included in
[`adapter/`](adapter/qwen2.5-1.5b-banking77-lora) (37 MB, via Git LFS) with its own model card,
so you can reproduce the 93.4% without retraining.

### What the numbers say

- **Fine-tuning closed a 57-point gap.** The base model zero-shot got 36.9% and produced invalid
  labels 6.5% of the time; after QLoRA it gets 93.4% with 0.1% invalid outputs.
- **A tuned linear model is a strong baseline.** TF-IDF + logistic regression is within 2.3
  points of the fine-tuned LLM at about 1/300th of the latency. For a closed label set with
  plenty of labelled data, try it first.
- **Prompting a frontier model underperformed both trained models here.** Banking77's labels
  encode annotation conventions (e.g. `card_arrival` vs `card_delivery_estimate`) that one
  example per label can't fully convey; trained models learn them from 9k examples. Claude
  would be the right choice with little or no labelled data, or when labels change often.
- **Cost/latency:** the prompted models are about 35-70x slower per query than the batched
  local adapter and cost ~$2 per 1K queries. Haiku 4.5 cost *more* than Opus 5 here: its
  4,096-token minimum cacheable prefix is above this 2.6K-token prompt, so every Haiku call
  paid full input price, while Opus 5 (512-token minimum) read the prompt from cache.

### Caveats

- Claude numbers come from a 300-query sample (95% CI roughly ±4 points), with effort `low` and
  one example per label. More shots or higher effort would likely help; that was not tuned.
- Claude costs use Anthropic list prices; Bedrock bills separately.
- LLM outputs are scored by exact label match (no fuzzy mapping), so invalid outputs count as wrong.
- Local latency is batched (batch 32) and hardware-specific; API latency is sequential per query.

## Method

- `data.py`: stratified 10% validation split carved from train (seed 42). All tuning (C values,
  early stopping) uses validation; test is only used for the final numbers.
- `baseline_sklearn.py`: word 1-2 gram + char 2-5 gram TF-IDF, logistic regression and linear
  SVM, C grid-searched on validation.
- `baseline_torch.py`: EmbeddingBag over unigrams + hashed bigrams, AdamW, early stopping on
  validation accuracy.
- `lora_llm.py`: 4-bit NF4 quantized base model; LoRA r=16 on all attention and MLP projections;
  TRL `SFTTrainer` with prompt/completion data so the loss is computed on the label tokens only.
- `claude_prompt.py`: label list + one train example per label in a system prompt with
  `cache_control`; cost computed from `usage` (input, cache write, cache read, output tokens).

## Project layout

```
src/finetune_intent_lab/
  data.py              Banking77 loading + fixed stratified validation split
  metrics.py           shared scoring (accuracy, macro-F1, invalid rate, latency) + MLflow logging
  baseline_sklearn.py  TF-IDF + logistic regression / linear SVM
  baseline_torch.py    EmbeddingBag classifier with a hand-written training loop
  lora_llm.py          zero-shot eval, QLoRA training and adapter eval for Qwen2.5-1.5B-Instruct
  claude_prompt.py     prompted Claude baseline on Amazon Bedrock, with prompt caching + cost tracking
adapter/qwen2.5-1.5b-banking77-lora/
                       trained LoRA adapter + tokenizer + model card (Git LFS)
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and all dependencies from `uv.lock`)
- [Git LFS](https://git-lfs.com/) to download the adapter weights (`git lfs install` once, before cloning)
- An NVIDIA GPU with ~4 GB free VRAM for the LLM steps (tested on an RTX 5050 Laptop, 8 GB,
  Windows 10). The PyTorch wheels are CUDA 12.8 builds, which also cover Blackwell (sm_120) GPUs.
  The scikit-learn and PyTorch baselines run on CPU.
- Optional: AWS credentials with Bedrock model access for the Claude baseline.

## Run it

```bash
git clone https://github.com/RahulRachhoya/finetune-intent-lab.git
cd finetune-intent-lab
uv sync                                               # Python 3.12, PyTorch CUDA 12.8 wheels
uv run python -m finetune_intent_lab.baseline_sklearn
uv run python -m finetune_intent_lab.baseline_torch
uv run python -m finetune_intent_lab.lora_llm zero-shot
uv run python -m finetune_intent_lab.lora_llm eval     # evaluate the included adapter (no training)
uv run python -m finetune_intent_lab.lora_llm train    # retrain (~18 min); overwrites adapter/
# needs AWS credentials with Bedrock access (AWS_PROFILE / AWS_REGION); costs ~$1 per model
uv run python -m finetune_intent_lab.claude_prompt --model claude-opus-5 --limit 300
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## License

MIT, see [LICENSE](LICENSE). Banking77 is released by PolyAI under CC BY 4.0; Qwen2.5 weights
are under the Apache 2.0 license.
