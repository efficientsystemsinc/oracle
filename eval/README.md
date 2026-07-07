# eval

`eval.sh` scores hit@5: each `probes.tsv` line is
`question<TAB>expected-regex[<TAB>miner[<TAB>as-of-date]]`, run through
`oracle query -k 5` against your live DB.

The committed `probes.tsv` is a SYNTHETIC example set derived from
`testdata/fixture.db` (see `scripts/make_synthetic_fixture.py`) — it only
scores meaningfully against that fixture. For your own DB, mine real probes
from your graph:

    oracle mineprobes > eval/probes.mine.tsv
    eval/eval.sh eval/probes.mine.tsv

Mined/personal probe files (`probes.mine*.tsv`, `probes_1k.tsv`,
`probes.*.tsv`, `ask_confidence_probes.json`) are gitignored: they contain
your session content — never commit them to a public fork.

`ask_confidence.md` documents the ask-confidence/abstain calibration method.
