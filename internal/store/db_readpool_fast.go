//go:build sqlite_fts5

package store

// db_readpool_fast.go — C-SQLite read pool (mattn/go-sqlite3), selected at
// BUILD time with `-tags sqlite_fts5` (the tag also compiles FTS5 into the C
// build — required, or the read pool fails loudly at open).
//
// Why: modernc (pure-Go) runs the bm25 FTS arms ~2.4x slower than C SQLite —
// it is the dominant residual cost of the query path after the GPU vector
// store. Same SQLite semantics (same FTS5, same bm25), only the engine build
// differs; the modernc writer pool keeps handling all writes/schema.
//
// Default builds (no tag) use the modernc read pool in db.go.

import (
	"database/sql"
	"fmt"

	_ "github.com/mattn/go-sqlite3"
)

const fastReadPool = true

// openReadPool opens the C-SQLite query_only pool over the same file.
func openReadPool(path string) (*sql.DB, error) {
	dsn := fmt.Sprintf("file:%s?_journal_mode=WAL&_busy_timeout=30000&_query_only=1", path)
	ro, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open C-sqlite read pool: %w", err)
	}
	ro.SetMaxOpenConns(4)
	return ro, nil
}
