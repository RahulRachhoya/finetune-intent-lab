---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
- mteb/banking77
language:
- en
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
- sft
- trl
- intent-classification
---

# Qwen2.5-1.5B-Instruct QLoRA adapter for Banking77 intent classification

A LoRA adapter that turns Qwen2.5-1.5B-Instruct into a generative intent classifier for
online-banking customer queries: given a query, it replies with one of the 77
[Banking77](https://huggingface.co/datasets/mteb/banking77) intent labels (e.g. `card_arrival`).

Trained and evaluated in [finetune-intent-lab](https://github.com/RahulRachhoya/finetune-intent-lab),
which compares it with scikit-learn, PyTorch and prompted-Claude baselines.

## Results (official Banking77 test set, 3,076 queries)

| Model | Accuracy | Macro-F1 | Invalid outputs |
|---|---|---|---|
| Base model, zero-shot (label list in prompt) | 36.9% | 0.341 | 6.5% |
| **This adapter** | **93.4%** | **0.935** | **0.1%** |

An output counts as invalid, and wrong, if it is not an exact label string.

## Training

| | |
|---|---|
| Method | QLoRA: base model loaded in 4-bit NF4 (double quantization, bf16 compute) |
| LoRA | r=16, alpha=32, dropout=0.05, on q/k/v/o and gate/up/down projections |
| Trainable parameters | 18.5M (~1.2% of the base model) |
| Data | Banking77 train minus a stratified 10% validation split (8,993 queries) |
| Objective | TRL `SFTTrainer`, prompt/completion format, loss on the label tokens only |
| Schedule | 2 epochs, batch 16, lr 2e-4, cosine decay, 30 warmup steps |
| Hardware | 1x RTX 5050 Laptop (8 GB); 18 minutes, 2.1 GB peak VRAM |

## Usage

The adapter expects the same prompt it was trained with:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "adapter/qwen2.5-1.5b-banking77-lora"   # path inside the cloned repo

tok = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(
    base, device_map={"": 0}, dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                           bnb_4bit_use_double_quant=True),
)
model = PeftModel.from_pretrained(model, adapter)

prompt = ("Classify this online-banking customer query into its intent label. "
          "Reply with the label only.\nQuery: I am still waiting on my card?")
inputs = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))   # card_arrival
```

## Limitations

- Only knows the 77 Banking77 intents; anything out of scope gets forced into one of them.
- English only, trained on short single-turn queries.
- Evaluated with 4-bit inference; results in full precision were not measured.

## License

Apache 2.0, following the Qwen2.5-1.5B-Instruct base model. Banking77 is CC BY 4.0 (PolyAI).
