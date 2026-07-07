#!/bin/bash
# Self-contained continuation on the TPU box: paraphrase -> export -> retrain
# embedder -> embed corpus+questions -> A/B vs the remote embedder. Survives laptop death.
set -x
cd ~/cloudrun
export ORACLE_HOME=~/cloudrun
: "${ORACLE_LLM_URL:?set ORACLE_LLM_URL/ORACLE_LLM_KEY before running}"
# 1. finish paraphrase coverage (resumable; loop through transport errors)
for i in $(seq 1 60); do
  ./oracle-linux paraphrase --max-calls 50 >> paraphrase.log 2>&1
  left=$(python3 -c "import sqlite3;db=sqlite3.connect('oracle.db');print(db.execute('SELECT (SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL)-(SELECT COUNT(DISTINCT fact_id) FROM fact_paraphrases)').fetchone()[0])")
  echo "left=$left" >> paraphrase.log
  [ "$left" -le 100 ] && break
  sleep 5
done
# 2. export training pairs: paraphrases + supersession-chain pairs
python3 - <<'PYEOF'
import sqlite3, json
db = sqlite3.connect('oracle.db')
n=0
with open('embed_pairs.jsonl','w') as f:
    for s,p in db.execute("SELECT f.statement, p.text FROM fact_paraphrases p JOIN facts f ON f.id=p.fact_id"):
        f.write(json.dumps({"statement":s,"paraphrase":p})+"\n"); n+=1
    for a,b in db.execute("""SELECT o.statement, n.statement FROM facts o JOIN facts n ON n.id=o.superseded_by
                             WHERE abs(n.valid_from-o.valid_from) < 21*86400"""):
        f.write(json.dumps({"statement":b,"paraphrase":a})+"\n"); n+=1
print("PAIRS", n)
PYEOF
# 3. retrain embedder (bigger data, 3 epochs, batch 256 for harder in-batch negatives)
python3 embedder_train.py --epochs 3 --batch 256 --out embedder_v2 > embedder_v2.log 2>&1
# 4. embed live corpus + probe questions with v2
python3 - <<'PYEOF'
import sqlite3, json
db = sqlite3.connect('oracle.db')
with open('live_facts.jsonl','w') as f:
    for i,s in db.execute("SELECT id, statement FROM facts WHERE superseded_at IS NULL"):
        f.write(json.dumps({"id":i,"statement":s})+"\n")
PYEOF
sed 's/embedder.msgpack/embedder_v2.msgpack/' embed_corpus.py > embed_corpus_v2.py
python3 embed_corpus_v2.py > embed_corpus_v2.log 2>&1
sed 's/embedder.msgpack/embedder_v2.msgpack/' embed_questions.py > embed_questions_v2.py
python3 embed_questions_v2.py > embed_questions_v2.log 2>&1
# 5. A/B vs the remote embedder on cosine-arm hit@5
python3 ab_eval.py > AB_RESULTS.txt 2>&1
echo CLOUD_DRIVER_DONE >> AB_RESULTS.txt
