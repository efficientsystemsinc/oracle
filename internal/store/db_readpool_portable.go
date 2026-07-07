//go:build !sqlite_fts5

package store

// db_readpool_portable.go — default read pool: modernc (pure Go), same driver
// as the writer. Build with `-tags sqlite_fts5` for the ~2.4x-faster C-SQLite
// read pool (see db_readpool_fast.go).

import (
	"database/sql"
	"fmt"
)

const fastReadPool = false

func openReadPool(path string) (*sql.DB, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=busy_timeout(30000)&_pragma=query_only(1)", path)
	ro, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open read pool: %w", err)
	}
	ro.SetMaxOpenConns(4)
	return ro, nil
}
