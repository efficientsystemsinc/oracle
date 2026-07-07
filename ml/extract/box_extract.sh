#!/bin/bash
# Cron-safe launcher: bootstrap a venv if needed, wait for extract_distill.jsonl, then run train_extract.py in the background.
# usage: box_extract.sh   (runs in ~/extract; idempotent — exits if already trained/running; logs to train.log)
cd ~/extract || exit 1
if [ ! -d venv ]; then python3 -m venv venv && ./venv/bin/pip -q install torch transformers peft accelerate safetensors 2>&1 | tail -1; fi
grep -q TRAIN_DONE train.log 2>/dev/null && exit 0
pgrep -f train_extract.py >/dev/null && exit 0
until [ -f extract_distill.jsonl ]; do sleep 60; done
setsid nohup ./venv/bin/python train_extract.py >> train.log 2>&1 &
