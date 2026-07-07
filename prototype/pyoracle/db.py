"""SQLite store. Bi-temporal: recorded_at/superseded_at = transaction time,
valid_from = world time. Superseded rows never deleted."""
import os
import sqlite3

ORACLE_HOME = os.path.expanduser(os.environ.get("ORACLE_HOME", "~/.oracle"))
DB_PATH = os.path.join(ORACLE_HOME, "oracle.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  path TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  repo TEXT,
  offset INTEGER NOT NULL DEFAULT 0,
  mtime REAL NOT NULL DEFAULT 0,
  last_scan REAL
);
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY,
  statement TEXT NOT NULL,
  kind TEXT NOT NULL,
  repo TEXT,
  entities TEXT NOT NULL DEFAULT '[]',
  files TEXT NOT NULL DEFAULT '[]',
  confidence REAL NOT NULL DEFAULT 0.7,
  mass REAL NOT NULL DEFAULT 1.0,
  recorded_at REAL NOT NULL,
  valid_from REAL NOT NULL,
  superseded_at REAL,
  superseded_by INTEGER,
  src_path TEXT,
  src_session TEXT,
  last_used_at REAL,
  use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_facts_repo ON facts(repo);
CREATE INDEX IF NOT EXISTS idx_facts_live ON facts(superseded_at);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  statement, entities, content='facts', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, statement, entities) VALUES (new.id, new.statement, new.entities);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, statement, entities) VALUES ('delete', old.id, old.statement, old.entities);
END;
CREATE TABLE IF NOT EXISTS edges(
  src INTEGER NOT NULL,
  dst INTEGER NOT NULL,
  type TEXT NOT NULL,
  recorded_at REAL NOT NULL,
  PRIMARY KEY(src, dst, type)
);
CREATE TABLE IF NOT EXISTS traces(
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  q TEXT,
  results TEXT
);
CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,      -- canonical: lowercased
  display TEXT NOT NULL,
  seen_count INTEGER NOT NULL DEFAULT 0,
  last_seen REAL
);
CREATE TABLE IF NOT EXISTS fact_entities(
  fact_id INTEGER NOT NULL,
  entity_id INTEGER NOT NULL,
  PRIMARY KEY(fact_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_fe_entity ON fact_entities(entity_id);
CREATE TABLE IF NOT EXISTS entity_edges(
  a INTEGER NOT NULL,
  b INTEGER NOT NULL,             -- a < b
  count INTEGER NOT NULL DEFAULT 0,
  last_seen REAL,
  PRIMARY KEY(a, b)
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


def connect() -> sqlite3.Connection:
    os.makedirs(ORACLE_HOME, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con
