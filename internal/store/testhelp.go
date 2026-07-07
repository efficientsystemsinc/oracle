package store

// Shared test fixtures used by tests across oracle's packages. Kept in a
// non-test file so other packages' tests can import them via package store.

import (
	"database/sql"
	"testing"
)

// TestDB opens a fresh oracle db under a temp ORACLE_HOME.
func TestDB(t *testing.T) *sql.DB {
	t.Helper()
	t.Setenv("ORACLE_HOME", t.TempDir())
	db, err := OpenDB()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// InsertFact inserts a minimal live fact and returns its id.
func InsertFact(t *testing.T, db *sql.DB, statement, kind, repo string, validFrom float64) int64 {
	t.Helper()
	res, err := db.Exec(`INSERT INTO facts(statement, kind, repo, recorded_at, valid_from)
		VALUES(?,?,?,?,?)`, statement, kind, repo, validFrom, validFrom)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := res.LastInsertId()
	return id
}

// Supersede closes oldID with newID at transaction time txAt.
func Supersede(t *testing.T, db *sql.DB, oldID, newID int64, txAt float64) {
	t.Helper()
	if _, err := db.Exec(`UPDATE facts SET superseded_at = ?, superseded_by = ? WHERE id = ?`,
		txAt, newID, oldID); err != nil {
		t.Fatal(err)
	}
}
