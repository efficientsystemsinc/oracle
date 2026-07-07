package kb

import (
	"database/sql"
	"testing"
)

func InsertEntity(t *testing.T, db *sql.DB, name string) int64 {
	t.Helper()
	res, err := db.Exec(`INSERT INTO entities(name, display, seen_count, last_seen, vkey)
		VALUES(?,?,1,0,?)`, name, name, VariantKey(name))
	if err != nil {
		t.Fatal(err)
	}
	id, _ := res.LastInsertId()
	return id
}
