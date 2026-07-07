package store

// Tests for backup.go.

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestBackupDBSnapshotAndRetention(t *testing.T) {
	db := TestDB(t)
	InsertFact(t, db, "backup me", "fact", "oracle", 100)
	InsertFact(t, db, "me too", "fact", "oracle", 200)

	// pre-seed 8 fake older snapshots; retention must fold to backupKeep total
	dir := filepath.Join(OracleHome(), "backups")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		old := filepath.Join(dir, fmt.Sprintf("oracle-2020010%d-000000.db", i+1))
		if err := os.WriteFile(old, []byte("stale"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	dest, err := BackupDB(db)
	if err != nil {
		t.Fatal(err)
	}

	// the snapshot is a valid sqlite db with the same facts
	snap, err := sql.Open("sqlite", "file:"+dest+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	var n int
	if err := snap.QueryRow("SELECT COUNT(*) FROM facts").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("snapshot has %d facts, want 2", n)
	}

	matches, _ := filepath.Glob(filepath.Join(dir, "oracle-*.db"))
	if len(matches) != backupKeep {
		t.Errorf("retention: %d snapshots remain, want %d", len(matches), backupKeep)
	}
	found := false
	for _, m := range matches {
		if m == dest {
			found = true
		}
	}
	if !found {
		t.Error("newest snapshot was pruned")
	}

	// immediately after a snapshot, none is due
	dest2, err := BackupIfDue(db)
	if err != nil {
		t.Fatal(err)
	}
	if dest2 != "" {
		t.Errorf("backup should not be due right after one, got %q", dest2)
	}
}

func TestPruneTraces(t *testing.T) {
	db := TestDB(t)
	for i := 0; i < 150; i++ {
		if _, err := db.Exec(`INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)`,
			float64(i), "query", fmt.Sprintf("q%d", i), "[]"); err != nil {
			t.Fatal(err)
		}
	}
	removed, err := PruneTraces(db, 100)
	if err != nil {
		t.Fatal(err)
	}
	if removed != 50 {
		t.Errorf("removed %d, want 50", removed)
	}
	var n, minID int
	if err := db.QueryRow("SELECT COUNT(*), MIN(id) FROM traces").Scan(&n, &minID); err != nil {
		t.Fatal(err)
	}
	if n != 100 || minID != 51 {
		t.Errorf("want newest 100 rows (min id 51), got %d rows min id %d", n, minID)
	}
	// under the cap: no-op
	if removed, err = PruneTraces(db, 100); err != nil || removed != 0 {
		t.Errorf("under cap should be a no-op, got %d %v", removed, err)
	}
}
