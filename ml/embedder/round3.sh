#!/bin/bash
# Embedder round 3: mix question-style pairs into embed_pairs.jsonl, retrain on e5-base-v2, re-embed corpus+questions, A/B eval vs Azure.
# usage: round3.sh   (runs in ~/cloudrun; waits for q_pairs.jsonl to appear; writes embedder_v3.msgpack + AB_V3_RESULTS.txt)
cd ~/cloudrun
until [ -f q_pairs.jsonl ]; do sleep 60; done
# round 3: retrieval-pretrained base + question-style pairs mixed with paraphrases
python3 - <<'PYEOF'
import json, random
pairs=[json.loads(l) for l in open('q_pairs.jsonl')]
old=[json.loads(l) for l in open('embed_pairs.jsonl')]
random.seed(7); random.shuffle(old)
mixed = pairs + old[:len(pairs)//2]   # questions dominate: they match query distribution
random.shuffle(mixed)
with open('embed_pairs.jsonl','w') as f:
    for d in mixed: f.write(json.dumps(d)+'\n')
print('mixed pairs', len(mixed))
PYEOF
python3 embedder_train.py --model intfloat/e5-base-v2 --epochs 3 --batch 256 --out embedder_v3 > embedder_v3.log 2>&1
sed 's/embedder.msgpack/embedder_v3.msgpack/; s/bert-base-uncased/intfloat\/e5-base-v2/' embed_corpus.py > embed_corpus_v3.py
python3 embed_corpus_v3.py > ec3.log 2>&1
sed 's/embedder.msgpack/embedder_v3.msgpack/; s/bert-base-uncased/intfloat\/e5-base-v2/' embed_questions.py > embed_questions_v3.py
python3 embed_questions_v3.py > eq3.log 2>&1
python3 ab_eval.py > AB_V3_RESULTS.txt 2>&1
echo ROUND3_DONE >> AB_V3_RESULTS.txt
