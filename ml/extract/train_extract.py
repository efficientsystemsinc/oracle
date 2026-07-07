"""LoRA fine-tune Qwen2.5-1.5B-Instruct on (chunk -> facts JSON) distillation.
GPU0 only. Prompt tokens masked; completion = teacher's JSON verbatim.

usage: python3 train_extract.py   (run in a dir with extract_distill.jsonl + extract_prompt.txt; writes extract_lora/, extract_merged/, extract_test.json)
"""
import json, os, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAXLEN = 8192
tok = AutoTokenizer.from_pretrained(BASE)
SYS = open("extract_prompt.txt").read()

rows = [json.loads(l) for l in open("extract_distill.jsonl")]
random.seed(7); random.shuffle(rows)
test = rows[:150]; train = rows[150:]
json.dump(test, open("extract_test.json","w"))
print("train", len(train), "test", len(test), flush=True)

def build(r):
    msgs = [{"role":"system","content":SYS},{"role":"user","content":r["input"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + r["output"] + tok.eos_token
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    fids = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"]
    labels = [-100]*min(len(pids),len(fids)) + fids[min(len(pids),len(fids)):]
    return {"input_ids": fids, "labels": labels}

ds = [build(r) for r in train]
ds = [d for d in ds if len(d["input_ids"]) > len([x for x in d["labels"] if x==-100])]  # must have completion tokens
print("usable", len(ds), flush=True)

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(ds)
    def __getitem__(self, i): return ds[i]

def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(batch), ml), pad, dtype=torch.long)
    lab = torch.full((len(batch), ml), -100, dtype=torch.long)
    att = torch.zeros((len(batch), ml), dtype=torch.long)
    for i,b in enumerate(batch):
        n=len(b["input_ids"])
        ids[i,:n]=torch.tensor(b["input_ids"]); lab[i,:n]=torch.tensor(b["labels"]); att[i,:n]=1
    return {"input_ids":ids,"labels":lab,"attention_mask":att}

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"":0})
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(output_dir="extract_lora", num_train_epochs=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=16, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    bf16=True, logging_steps=20, save_strategy="epoch", report_to=[])
Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
model.save_pretrained("extract_lora/final")
# merge for export
merged = model.merge_and_unload()
merged.save_pretrained("extract_merged", safe_serialization=True)
tok.save_pretrained("extract_merged")
print("TRAIN_DONE")
