#!/bin/bash
# hit@5: does any of the top-5 statements match the expected regex?
# usage: eval.sh [probes.tsv]  (default: probes.tsv next to this script)
# TSV columns: question<TAB>regex[<TAB>miner[<TAB>as-of-date]]
# Columns 3-4 are optional: miner tags rows for the per-miner breakdown; a
# non-empty 4th column runs the probe as `oracle query --as-of <date>`.
probes="${1:-$(dirname "$0")/probes.tsv}"
hits=0; total=0
declare -A mhits mtotal
while IFS=$'\t' read -r q expect miner asof; do
  [ -z "$q" ] && continue
  miner="${miner:-untagged}"
  total=$((total+1)); mtotal[$miner]=$(( ${mtotal[$miner]:-0} + 1 ))
  if [ -n "$asof" ]; then
    out=$(oracle query "$q" -k 5 --as-of "$asof" 2>/dev/null)
  else
    out=$(oracle query "$q" -k 5 2>/dev/null)
  fi
  if echo "$out" | grep -qiE "$expect"; then
    hits=$((hits+1)); mhits[$miner]=$(( ${mhits[$miner]:-0} + 1 ))
    echo "HIT  [$miner${asof:+ as-of $asof}] $q"
  else
    echo "MISS [$miner${asof:+ as-of $asof}] $q  (want: $expect)"
  fi
done < "$probes"
echo "----"
echo "hit@5: $hits/$total"
for m in "${!mtotal[@]}"; do
  echo "  $m: ${mhits[$m]:-0}/${mtotal[$m]}"
done
