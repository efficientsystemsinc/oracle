package store

// Tests for stmthash.go.

import "testing"

func TestNormalizeStatementCollisions(t *testing.T) {
	// formatting variants MUST collide
	collide := [][2]string{
		{"Quasar API runs on atlas01.", "quasar api runs on atlas01"},
		{"recall@10 = 0.58 on eval_v2", "Recall @ 10 = 0.58 on eval-v2"},
		{"cost is 1,234 USD", "cost is 1234 USD"},
		{"pgbouncer:  txn   pooling", "pgbouncer txn pooling"},
		{"deploy.sh clobbers box hot-fixes", "deploy sh clobbers box hotfixes"},
	}
	for _, p := range collide {
		if normalizeStatement(p[0]) != normalizeStatement(p[1]) {
			t.Errorf("should collide: %q vs %q (%q != %q)", p[0], p[1],
				normalizeStatement(p[0]), normalizeStatement(p[1]))
		}
	}
	// digits are semantic: different numbers MUST NOT collide
	distinct := [][2]string{
		{"recall 0.58 on eval_v2", "recall 0.72 on eval_v2"},
		{"pool has 4 replicas", "pool has 11 replicas"},
		{"cost is 1234 USD", "cost is 12345 USD"},
		{"quasar api runs on atlas01", "quasar api runs on c8z4"},
		{"latency p95 6.0s", "latency p50 6.0s"},
	}
	for _, p := range distinct {
		if normalizeStatement(p[0]) == normalizeStatement(p[1]) {
			t.Errorf("must NOT collide: %q vs %q (both %q)", p[0], p[1], normalizeStatement(p[0]))
		}
	}
	if StmtHash("Recall: 0.58!") != StmtHash("recall 0.58") {
		t.Error("stmtHash should be formatting-invariant")
	}
	if StmtHash("recall 0.58") == StmtHash("recall 0.72") {
		t.Error("stmtHash must distinguish digit values")
	}
}

func TestStmtHashBackfill(t *testing.T) {
	db := TestDB(t)
	// simulate a pre-column row: insert then null the hash
	f := InsertFact(t, db, "Quasar API runs on atlas01.", "fact", "quasar", 100)
	if _, err := db.Exec("UPDATE facts SET stmt_hash = NULL WHERE id = ?", f); err != nil {
		t.Fatal(err)
	}
	if err := backfillStmtHashes(db); err != nil {
		t.Fatal(err)
	}
	var h string
	if err := db.QueryRow("SELECT stmt_hash FROM facts WHERE id = ?", f).Scan(&h); err != nil {
		t.Fatal(err)
	}
	if h != StmtHash("Quasar API runs on atlas01.") {
		t.Errorf("backfill hash mismatch: %q", h)
	}
}
