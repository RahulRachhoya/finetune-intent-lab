"""QLoRA fine-tune of an open-weight instruct model as a generative intent classifier.

The model reads a query and generates the intent name (e.g. "card_arrival"). Outputs that are
not an exact label count as invalid (-1) rather than being fuzzy-matched, so the score is honest.

    python -m finetune_intent_lab.lora_llm zero-shot   # base model, label list in the prompt
    python -m finetune_intent_lab.lora_llm train       # QLoRA SFT, then evaluate the adapter
"""
import argparse
import time
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from finetune_intent_lab import data, metrics

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path("adapter/qwen2.5-1.5b-banking77-lora")
INSTRUCTION = "Classify this online-banking customer query into its intent label. Reply with the label only."


def quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )


def user_message(text: str, label_names: list[str] | None) -> str:
    msg = INSTRUCTION
    if label_names:  # zero-shot: the model has never seen the label set, so list it
        msg += "\nValid labels: " + ", ".join(label_names)
    return f"{msg}\nQuery: {text}"


def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config(), dtype=torch.bfloat16, device_map={"": 0},
    )
    return tok, model


@torch.no_grad()
def predict(tok, model, texts, label_names, list_labels: bool, batch_size: int) -> list[int]:
    model.eval()
    index = {name: i for i, name in enumerate(label_names)}
    preds = []
    for i in range(0, len(texts), batch_size):
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": user_message(t, label_names if list_labels else None)}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in texts[i:i + batch_size]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for seq in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            preds.append(index.get(seq.strip(), -1))
        print(f"  predicted {len(preds)}/{len(texts)}", end="\r")
    print()
    return preds


def evaluate(tok, model, ds, list_labels: bool, batch_size: int):
    for split in ("val", "test"):
        s = getattr(ds, split)
        t = time.perf_counter()
        pred = predict(tok, model, s.texts, ds.label_names, list_labels, batch_size)
        metrics.log_split(split, s.labels, pred, time.perf_counter() - t)


def zero_shot(args):
    ds = data.load()
    tok, model = load_base()
    with metrics.start_run("qwen2.5-1.5b-zero-shot"):
        mlflow.log_params({"approach": "llm_zero_shot", "model": BASE_MODEL, "quant": "nf4-4bit"})
        evaluate(tok, model, ds, list_labels=True, batch_size=args.eval_batch)


def to_sft(split: data.Split, label_names) -> Dataset:
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": user_message(t, None)}],
         "completion": [{"role": "assistant", "content": label_names[y]}]}
        for t, y in zip(split.texts, split.labels)
    ])


def train(args):
    ds = data.load()
    tok, model = load_base()
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir="outputs/sft-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,  # ~3% of 2 epochs at batch 16
        bf16=True,
        gradient_checkpointing=True,
        max_length=128,
        logging_steps=20,
        eval_strategy="epoch",
        per_device_eval_batch_size=32,
        save_strategy="no",
        report_to="mlflow",
        seed=data.SEED,
    )
    with metrics.start_run("qwen2.5-1.5b-qlora"):
        mlflow.log_params({"approach": "llm_qlora", "model": BASE_MODEL, "quant": "nf4-4bit",
                           "lora_rank": args.rank, "lora_targets": "all linear"})
        trainer = SFTTrainer(
            model=model, args=cfg, peft_config=lora, processing_class=tok,
            train_dataset=to_sft(ds.train, ds.label_names), eval_dataset=to_sft(ds.val, ds.label_names),
        )
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        mlflow.log_param("trainable_params", trainable)
        t = time.perf_counter()
        trainer.train()
        mlflow.log_metrics({"train_seconds": time.perf_counter() - t,
                            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
        trainer.model.save_pretrained(ADAPTER_DIR)
        tok.save_pretrained(ADAPTER_DIR)
        evaluate(tok, trainer.model, ds, list_labels=False, batch_size=args.eval_batch)


def evaluate_saved(args):
    ds = data.load()
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    with metrics.start_run("qwen2.5-1.5b-qlora-eval"):
        mlflow.log_params({"approach": "llm_qlora", "model": BASE_MODEL, "adapter": str(ADAPTER_DIR)})
        evaluate(tok, model, ds, list_labels=False, batch_size=args.eval_batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["zero-shot", "train", "eval"])
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    args = p.parse_args()
    {"zero-shot": zero_shot, "train": train, "eval": evaluate_saved}[args.mode](args)


if __name__ == "__main__":
    main()
