"""Session watchers: tail Claude Code + codex jsonl files from stored byte
offsets, render new content into plain-text transcript chunks."""
import glob
import json
import os
import re
import time
from dataclasses import dataclass

CLAUDE_DIR = os.path.expanduser("~/.claude/projects")
CODEX_DIR = os.path.expanduser("~/.codex/sessions")
CHUNK_CHARS = 24_000
MAX_TOOL_CHARS = 400

# ponytail: bench sandboxes under /private/tmp are huge and low-signal; skip
SKIP_PROJECT_PREFIXES = ("-private-tmp", "-tmp-", "-var-")

SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|xox[bap]-[A-Za-z0-9-]{10,}"
    r"|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{40,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}"
    r"|(?i:(password|api[_-]?key|secret|token)\s*[=:]\s*)[^\s'\"]{8,})"
)


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


@dataclass
class Chunk:
    path: str
    source: str
    repo: str
    session: str
    text: str
    end_offset: int
    event_time: float


def _repo_from_cwd(cwd: str | None) -> str:
    if not cwd:
        return "unknown"
    return os.path.basename(cwd.rstrip("/")) or "unknown"


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "text":
                parts.append(c.get("text", ""))
            elif t == "tool_use":
                inp = json.dumps(c.get("input", {}))[:MAX_TOOL_CHARS]
                parts.append(f"[tool:{c.get('name')}] {inp}")
            elif t == "tool_result":
                raw = c.get("content")
                s = raw if isinstance(raw, str) else json.dumps(raw)[:MAX_TOOL_CHARS]
                parts.append(f"[result] {s[:MAX_TOOL_CHARS]}")
    return "\n".join(p for p in parts if p)


def _parse_claude_line(rec: dict) -> tuple[str, str | None, float | None]:
    """-> (rendered text or '', cwd, event_ts)"""
    t = rec.get("type")
    ts = None
    if rec.get("timestamp"):
        try:
            ts = _iso_to_epoch(rec["timestamp"])
        except ValueError:
            ts = None
    if t not in ("user", "assistant"):
        return "", None, ts
    msg = rec.get("message") or {}
    text = _content_text(msg.get("content"))
    if not text.strip():
        return "", rec.get("cwd"), ts
    role = "USER" if t == "user" else "ASSISTANT"
    if rec.get("isSidechain"):
        role = "SUBAGENT-" + role
    return f"{role}: {text}", rec.get("cwd"), ts


def _parse_codex_line(rec: dict) -> tuple[str, str | None, float | None]:
    ts = None
    if rec.get("timestamp"):
        try:
            ts = _iso_to_epoch(rec["timestamp"])
        except ValueError:
            ts = None
    p = rec.get("payload") or {}
    if rec.get("type") == "session_meta":
        return "", p.get("cwd"), ts
    if rec.get("type") != "response_item":
        return "", None, ts
    pt = p.get("type")
    if pt == "message":
        texts = [c.get("text", "") for c in p.get("content", []) if isinstance(c, dict)]
        body = "\n".join(x for x in texts if x)
        role = (p.get("role") or "assistant").upper()
        return (f"{role}: {body}" if body.strip() else ""), None, ts
    if pt == "function_call":
        args = (p.get("arguments") or "")[:MAX_TOOL_CHARS]
        return f"ASSISTANT: [tool:{p.get('name')}] {args}", None, ts
    return "", None, ts


def _iso_to_epoch(s: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


PARSERS = {"claude": _parse_claude_line, "codex": _parse_codex_line}


def discover(since_days: float | None = None) -> list[tuple[str, str]]:
    """-> [(path, source)] of session files worth scanning."""
    out = []
    cutoff = time.time() - since_days * 86400 if since_days else None
    for d in sorted(glob.glob(os.path.join(CLAUDE_DIR, "*"))):
        base = os.path.basename(d)
        if base.startswith(SKIP_PROJECT_PREFIXES):
            continue
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            if cutoff and os.path.getmtime(f) < cutoff:
                continue
            out.append((f, "claude"))
    for f in glob.glob(os.path.join(CODEX_DIR, "**", "*.jsonl"), recursive=True):
        if cutoff and os.path.getmtime(f) < cutoff:
            continue
        out.append((f, "codex"))
    return out


def read_new(path: str, source: str, offset: int, known_repo: str | None) -> list[Chunk]:
    """Read from offset to EOF, render new complete lines into chunks.
    Returns [] if nothing new. end_offset only advances past complete lines."""
    size = os.path.getsize(path)
    if size <= offset:
        return []
    parse = PARSERS[source]
    session = os.path.splitext(os.path.basename(path))[0]
    repo = known_repo or "unknown"
    pieces: list[str] = []
    chunks: list[Chunk] = []
    last_ts = os.path.getmtime(path)
    with open(path, "rb") as fh:
        fh.seek(offset)
        pos = offset
        acc = 0
        for raw in fh:
            if not raw.endswith(b"\n"):
                break  # partial trailing line; pick it up next cycle
            pos += len(raw)
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text, cwd, ts = parse(rec)
            if cwd:
                repo = _repo_from_cwd(cwd)
            if ts:
                last_ts = ts
            if not text:
                continue
            text = redact(text)
            pieces.append(text)
            acc += len(text)
            if acc >= CHUNK_CHARS:
                chunks.append(Chunk(path, source, repo, session, "\n\n".join(pieces), pos, last_ts))
                pieces, acc = [], 0
    if pieces:
        chunks.append(Chunk(path, source, repo, session, "\n\n".join(pieces), pos, last_ts))
    elif not chunks and pos > offset:
        # only non-message lines consumed; still advance the offset
        chunks.append(Chunk(path, source, repo, session, "", pos, last_ts))
    return chunks
