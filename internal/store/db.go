package store

// SQLite store. Bi-temporal: recorded_at/superseded_at = transaction time,
// valid_from = world time. Superseded rows never deleted.

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	_ "modernc.org/sqlite"
)

// roPools maps a writer *sql.DB to its query_only read pool. WAL gives one
// writer + many readers; the writer pool is capped at 1 connection, so
// search()'s concurrent read arms would serialize on it. readDB(db) returns
// the read pool when one was registered, else db itself (fixture DBs opened
// elsewhere are single-caller and correct either way).
var roPools sync.Map

func registerReadPool(db, ro *sql.DB) { roPools.Store(db, ro) }

func ReadDB(db *sql.DB) *sql.DB {
	if v, ok := roPools.Load(db); ok {
		return v.(*sql.DB)
	}
	return db
}

const schema = `
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
-- search() hydration counts live contradictions per hit via a correlated
-- subquery on edges.dst; without this index that is a full edges scan per
-- hydrated fact (~1.6s/query at 12k edges — the dominant query-path cost).
CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, type);
CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
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
  b INTEGER NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  last_seen REAL,
  PRIMARY KEY(a, b)
);
CREATE TABLE IF NOT EXISTS traces(
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  q TEXT,
  results TEXT
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
`

// dbq is the query surface shared by *sql.DB and *sql.Tx, so entity helpers
// work inside and outside transactions.
type DBQ interface {
	Exec(query string, args ...any) (sql.Result, error)
	Query(query string, args ...any) (*sql.Rows, error)
	QueryRow(query string, args ...any) *sql.Row
}

// asOfPredicate is the world-time visibility rule for --as-of queries, against
// facts aliased as f. Bind the as-of timestamp twice. A fact is visible at T
// iff it had already become true, and nothing that replaced it had become true
// yet. See docs/facts.md.
const AsOfPredicate = `f.valid_from <= ? AND (f.superseded_by IS NULL OR
	(SELECT s.valid_from FROM facts s WHERE s.id = f.superseded_by) > ?)`

func OracleHome() string {
	if h := os.Getenv("ORACLE_HOME"); h != "" {
		return h
	}
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	return filepath.Join(home, ".oracle")
}

func DBPath() string { return filepath.Join(OracleHome(), "oracle.db") }

func OpenDB() (*sql.DB, error) {
	if err := os.MkdirAll(OracleHome(), 0o755); err != nil {
		return nil, err
	}
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=busy_timeout(30000)", DBPath())
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1) // sqlite writer; serialize and avoid SQLITE_BUSY
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	// WAL allows one writer + many concurrent readers. The writer pool above
	// stays capped at 1; search()'s parallel read arms go through a second,
	// query_only pool on the same file (see readDB; db_readpool_*.go picks
	// the engine — C SQLite with -tags sqlite_fts5, else modernc).
	ro, err := openReadPool(DBPath())
	if err != nil {
		return nil, err
	}
	if _, err := ro.Exec(`SELECT rowid FROM facts_fts WHERE facts_fts MATCH '"zz-readpool-probe"' LIMIT 1`); err != nil {
		return nil, fmt.Errorf("read pool probe (FTS5 must be compiled in): %w", err)
	}
	registerReadPool(db, ro)
	if _, err := db.Exec(embedSchema); err != nil {
		return nil, fmt.Errorf("apply embed schema: %w", err)
	}
	if _, err := db.Exec(localEmbedSchema); err != nil {
		return nil, fmt.Errorf("apply local embed schema: %w", err)
	}
	if err := applyKBSchema(db); err != nil {
		return nil, err
	}
	if err := applyParaSchema(db); err != nil {
		return nil, err
	}
	return db, nil
}
