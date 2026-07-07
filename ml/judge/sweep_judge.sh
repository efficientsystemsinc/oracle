#!/bin/bash
# Judge hyperparameter sweep: 4 configs (model x epochs x lr x max-len) through judge_train.py, printing the eval lines.
# usage: sweep_judge.sh   (runs in ~/judge; needs judge_train.py + its judge_train.jsonl/judge_test.jsonl inputs)
cd ~/judge
for cfg in "bert-base-uncased 6 2e-5 384 b6-len384" "bert-base-uncased 6 3e-5 256 b6-lr3" "bert-large-uncased 4 1e-5 256 large-lr1" "bert-large-uncased 4 2e-5 384 large-len384"; do
  set -- $cfg
  echo "=== $5 ($1 ep=$2 lr=$3 len=$4)"
  python3 judge_train.py --model $1 --epochs $2 --lr $3 --max-len $4 --out judge_$5 2>/dev/null | grep -E '^epoch|BEST|binary'
done
echo SWEEP_DONE
