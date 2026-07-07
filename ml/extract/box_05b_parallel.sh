#!/bin/bash
# Cron-safe launcher: derive a Qwen2.5-0.5B variant of train_extract.py (GPU1) and run it in the background once extract_distill.jsonl exists.
# usage: box_05b_parallel.sh   (runs in ~/extract; idempotent — exits if already trained/running; logs to train05.log)
cd ~/extract
grep -q TRAIN_DONE train05.log 2>/dev/null && exit 0
pgrep -f "train_extract_05.py" >/dev/null && exit 0
[ -f extract_distill.jsonl ] || exit 0
sed 's|Qwen/Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-0.5B-Instruct|; s|extract_lora|extract_lora05|g; s|extract_merged|extract_merged05|; s|"CUDA_VISIBLE_DEVICES", "0"|"CUDA_VISIBLE_DEVICES", "1"|' train_extract.py > train_extract_05.py
setsid nohup ./venv/bin/python train_extract_05.py >> train05.log 2>&1 &
