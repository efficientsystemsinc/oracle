"""LLM extraction: transcript chunk -> durable facts (OpenAI-compatible endpoint).
Fails loudly — no fallback provider, no empty-list-on-error."""
import json
import os
import urllib.request

ENDPOINT = os.environ["ORACLE_LLM_URL"]

SYSTEM = """You extract durable memory from AI coding-agent session transcripts for a team knowledge graph called oracle.

Extract only facts worth remembering NEXT WEEK: decisions made (and why), hard-won gotchas, infrastructure/config truths, benchmark numbers, user preferences/workflow rules, project status changes, and standing TODOs. Skip: transient debugging noise, file-by-file edit narration, anything true only for this one session, generic knowledge.

Each fact must be a standalone atomic statement understandable with zero session context — name the systems, repos, boxes, numbers explicitly. Convert relative dates to absolute when the session date is given.

Return JSON: {"facts":[{"statement":str, "kind":"decision|fact|gotcha|preference|status|todo", "entities":[str, canonical short names of systems/repos/boxes/tools mentioned], "files":[repo-relative paths if central], "confidence":0.0-1.0}]}
Return {"facts":[]} if nothing durable. Max 12 facts per chunk; prefer fewer, denser."""

JUDGE = """You maintain a fact graph. For each NEW fact below, decide if it SUPERSEDES (replaces/outdates) any of the listed OLD facts — same topic, newer state. Duplicate ≈ same claim → supersede. Different aspects of the same system → no.
Return JSON: {"verdicts":[{"new_idx":int, "supersedes":[old_id,...]}]} with an entry only when supersedes is non-empty."""


def _key() -> str:
    return os.environ["ORACLE_LLM_KEY"]


def _chat(system: str, user: str, max_tokens: int = 6000) -> dict:
    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"api-key": _key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    content = choice["message"]["content"]
    if not content:
        raise RuntimeError(f"empty completion (finish_reason={choice.get('finish_reason')})")
    return json.loads(content)


VALID_KINDS = {"decision", "fact", "gotcha", "preference", "status", "todo"}


def extract_facts(chunk_text: str, repo: str, session_date: str) -> list[dict]:
    user = f"Repo: {repo}\nSession date: {session_date}\n\nTRANSCRIPT:\n{chunk_text}"
    out = _chat(SYSTEM, user)
    facts = out.get("facts")
    if not isinstance(facts, list):
        raise RuntimeError(f"bad extraction shape: {out}")
    clean = []
    for f in facts:
        if not isinstance(f, dict) or not f.get("statement"):
            continue
        kind = f.get("kind") if f.get("kind") in VALID_KINDS else "fact"
        clean.append({
            "statement": str(f["statement"]).strip(),
            "kind": kind,
            "entities": [str(e) for e in f.get("entities", []) if e][:12],
            "files": [str(p) for p in f.get("files", []) if p][:12],
            "confidence": min(1.0, max(0.0, float(f.get("confidence", 0.7)))),
        })
    return clean


def judge_supersede(new_facts: list[dict], candidates: dict[int, list[dict]]) -> dict[int, list[int]]:
    """candidates: new_idx -> [{'id':.., 'statement':..}]. -> new_idx -> [old ids]"""
    if not candidates:
        return {}
    lines = []
    for idx, olds in candidates.items():
        lines.append(f"NEW {idx}: {new_facts[idx]['statement']}")
        for o in olds:
            lines.append(f"  OLD {o['id']}: {o['statement']}")
    out = _chat(JUDGE, "\n".join(lines), max_tokens=2000)
    result: dict[int, list[int]] = {}
    for v in out.get("verdicts", []):
        try:
            idx, olds = int(v["new_idx"]), [int(x) for x in v.get("supersedes", [])]
        except (KeyError, TypeError, ValueError):
            continue
        allowed = {o["id"] for o in candidates.get(idx, [])}
        olds = [o for o in olds if o in allowed]
        if idx in candidates and olds:
            result[idx] = olds
    return result
