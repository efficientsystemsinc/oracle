package store

import (
	"database/sql"
	"fmt"
	"strings"
)

const embedSchema = `
CREATE TABLE IF NOT EXISTS fact_vecs(
  fact_id INTEGER PRIMARY KEY,
  vec BLOB NOT NULL
);`

const localEmbedSchema = `
CREATE TABLE IF NOT EXISTS fact_vecs_local(
  fact_id INTEGER PRIMARY KEY,
  vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_para_vecs_local(
  fact_id INTEGER PRIMARY KEY,
  vec BLOB NOT NULL
);`

const kbSchema = `
ALTER TABLE entities ADD COLUMN etype TEXT;
CREATE TABLE IF NOT EXISTS entity_aliases(
  id INTEGER PRIMARY KEY,
  entity_id INTEGER NOT NULL,
  alias TEXT NOT NULL,
  valid_from REAL NOT NULL,
  valid_to REAL,
  recorded_at REAL NOT NULL,
  confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_alias_lookup ON entity_aliases(alias) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases(entity_id);
CREATE TABLE IF NOT EXISTS predicates(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,          -- canonical: runs_on, deployed_to, part_of, uses, ...
  seen_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS predicate_aliases(
  alias TEXT PRIMARY KEY,
  predicate_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS triples(
  id INTEGER PRIMARY KEY,
  fact_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,        -- entities.id
  predicate_id INTEGER NOT NULL,
  object_id INTEGER,                  -- entities.id, or NULL when literal
  object_literal TEXT,
  recorded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_triples_subj ON triples(subject_id, predicate_id);
CREATE INDEX IF NOT EXISTS idx_triples_obj ON triples(object_id);
CREATE TABLE IF NOT EXISTS metrics(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,          -- canonical: pass_at_1, recall_at_10, cost_usd, latency_p95_s
  unit TEXT,
  seen_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS metric_observations(
  id INTEGER PRIMARY KEY,
  metric_id INTEGER NOT NULL,
  entity_id INTEGER,                  -- what the number is about
  value REAL NOT NULL,
  occurred_at REAL NOT NULL,          -- world time
  fact_id INTEGER,                    -- provenance
  recorded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_metric ON metric_observations(metric_id, occurred_at);
CREATE TABLE IF NOT EXISTS enrich_done(fact_id INTEGER PRIMARY KEY);
ALTER TABLE files ADD COLUMN error_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE facts ADD COLUMN corroborations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE facts ADD COLUMN quote TEXT;
CREATE TABLE IF NOT EXISTS chunk_log(
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  path TEXT, source TEXT, repo TEXT,
  chars INTEGER NOT NULL,
  n_facts INTEGER NOT NULL,
  text TEXT NOT NULL
);
ALTER TABLE facts ADD COLUMN evidence TEXT NOT NULL DEFAULT 'asserted';
ALTER TABLE files ADD COLUMN last_error_ts REAL NOT NULL DEFAULT 0;
ALTER TABLE entities ADD COLUMN vkey TEXT;
CREATE INDEX IF NOT EXISTS idx_entities_vkey ON entities(vkey);
ALTER TABLE facts ADD COLUMN stmt_hash TEXT;
CREATE TABLE IF NOT EXISTS referee_done(
  src INTEGER NOT NULL,
  dst INTEGER NOT NULL,
  verdict TEXT NOT NULL,
  decided_at REAL NOT NULL,
  PRIMARY KEY(src, dst)
);
CREATE INDEX IF NOT EXISTS idx_facts_stmt_hash ON facts(stmt_hash);
`

func applyKBSchema(db *sql.DB) error {
	for _, stmt := range strings.Split(kbSchema, ";") {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}
		if _, err := db.Exec(stmt); err != nil {
			if strings.Contains(err.Error(), "duplicate column name") {
				continue // column already added on a previous run
			}
			return fmt.Errorf("kb schema: %w", err)
		}
	}
	// backfill variant keys for rows that predate the vkey column;
	// SQL mirror of variantKey (names are already canon-lowercased)
	if _, err := db.Exec(`UPDATE entities SET vkey =
		replace(replace(replace(replace(lower(name),'-',''),'_',''),'.',''),' ','')
		WHERE vkey IS NULL`); err != nil {
		return fmt.Errorf("vkey backfill: %w", err)
	}
	// backfill formatting-invariant statement hashes for rows that predate
	// the stmt_hash column (near-dupe detection across parallel subagents)
	if err := backfillStmtHashes(db); err != nil {
		return fmt.Errorf("stmt_hash backfill: %w", err)
	}
	return nil
}

const paraSchema = `
CREATE TABLE IF NOT EXISTS fact_paraphrases(
  fact_id INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fp_fact ON fact_paraphrases(fact_id);
CREATE TABLE IF NOT EXISTS paraphrase_done(
  fact_id INTEGER PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS paraphrase_skip(
  fact_id INTEGER PRIMARY KEY,
  err TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_para_vecs(
  fact_id INTEGER PRIMARY KEY,
  vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS para_fts USING fts5(
  text, fact_id UNINDEXED, tokenize='porter unicode61'
);`

func applyParaSchema(db *sql.DB) error {
	for _, stmt := range strings.Split(paraSchema, ";") {
		if strings.TrimSpace(stmt) == "" {
			continue
		}
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("apply paraphrase schema: %w", err)
		}
	}
	return nil
}
