#!/bin/bash
# Judge OOD evals: temporal-split retrain, repo-split retrain, and cross-generator eval of judge_v2 on gpt-5.4 synth pairs.
# usage: ood_run.sh   (runs in ~/judge; needs judge_temporal_*/judge_repoood_* jsonl splits, waits for judge_synth54_test.jsonl; appends to OOD_RESULTS.txt)
cd ~/judge
set -x
# temporal OOD retrain: train on past, test on newest 10%
cp judge_temporal_train.jsonl judge_train.jsonl; cp judge_temporal_test.jsonl judge_test.jsonl
python3 judge_train.py --epochs 5 --lr 3e-5 --out judge_temporal 2>/dev/null | grep -E 'BEST|binary' > OOD_RESULTS.txt
# repo OOD retrain: train non-quasar, test quasar
cp judge_repoood_train.jsonl judge_train.jsonl; cp judge_repoood_test.jsonl judge_test.jsonl
python3 judge_train.py --epochs 5 --lr 3e-5 --out judge_repoood 2>/dev/null | grep -E 'BEST|binary' >> OOD_RESULTS.txt
# cross-generator: v2 model on gpt-5.4-generated clean pairs (wait for upload)
until [ -f judge_synth54_test.jsonl ]; do sleep 60; done
cp judge_train_real_backup.jsonl judge_train.jsonl 2>/dev/null || true
python3 judge_eval.py judge_v2.msgpack judge_synth54_test.jsonl >> OOD_RESULTS.txt 2>&1
echo OOD_DONE >> OOD_RESULTS.txt
