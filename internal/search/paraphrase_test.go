package search

// Tests for paraphrase.go.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"oracle/internal/store"
	"strings"
	"testing"
	"time"
)

// stubEmbedServer serves the Azure embeddings shape, returning vecFor(text).
func stubEmbedServer(t *testing.T, vecFor func(text string) []float32) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("stub decode: %v", err)
			http.Error(w, err.Error(), 400)
			return
		}
		type d struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		out := struct {
			Data []d `json:"data"`
		}{}
		for i, s := range req.Input {
			out.Data = append(out.Data, d{i, vecFor(s)})
		}
		_ = json.NewEncoder(w).Encode(out)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("ORACLE_EMBED_URL", srv.URL)
	t.Setenv("ORACLE_EMBED_KEY", "test-key")
}

func unitVec(hot int) []float32 {
	v := make([]float32, embedDims)
	v[hot] = 1
	return v
}

// addParaphrase hand-inserts a paraphrase row the way paraphraseRun does.
func addParaphrase(t *testing.T, db store.DBQ, factID int64, text string, vec []float32) {
	t.Helper()
	if _, err := db.Exec("INSERT INTO fact_paraphrases(fact_id, text) VALUES(?,?)", factID, text); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("INSERT INTO para_fts(text, fact_id) VALUES(?,?)", text, factID); err != nil {
		t.Fatal(err)
	}
	if vec != nil {
		if _, err := db.Exec("INSERT OR REPLACE INTO fact_para_vecs(fact_id, vec) VALUES(?,?)", factID, store.VecToBlob(vec)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec("INSERT OR IGNORE INTO paraphrase_done(fact_id) VALUES(?)", factID); err != nil {
		t.Fatal(err)
	}
}

// TestSearchParaphraseFTSArm: a query worded like the paraphrase (zero lexical
// overlap with the canonical statement) must retrieve the fact via para_fts.
func TestSearchParaphraseFTSArm(t *testing.T) {
	db := store.TestDB(t)
	// zero query vector => cosine arm is inert; only lexical arms rank
	stubEmbedServer(t, func(string) []float32 { return make([]float32, embedDims) })

	now := float64(time.Now().Unix())
	target := store.InsertFact(t, db, "kubelet rotates serving certificates automatically", "fact", "infra", now)
	decoy := store.InsertFact(t, db, "grafana dashboard uses prometheus datasource", "fact", "infra", now)
	addParaphrase(t, db, target, "TLS cert renewal happens on its own for worker daemons", nil)

	hits, err := Search(db, "TLS cert renewal worker daemons", "", 5, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].ID != target {
		t.Fatalf("paraphrase FTS arm miss: hits=%+v want top=%d", hits, target)
	}
	for _, h := range hits {
		if h.ID == decoy {
			t.Fatalf("decoy retrieved for paraphrase-only query: %+v", hits)
		}
	}
}

// TestSearchParaphraseCosineMerge: paraphrase vector closer to the query than
// the canonical statement vector must lift the fact in the cosine arm.
func TestSearchParaphraseCosineMerge(t *testing.T) {
	db := store.TestDB(t)
	stubEmbedServer(t, func(string) []float32 { return unitVec(0) })

	now := float64(time.Now().Unix())
	a := store.InsertFact(t, db, "alpha statement", "fact", "", now)
	b := store.InsertFact(t, db, "beta statement", "fact", "", now)
	// canonical vecs: b slightly aligned, a orthogonal
	if _, err := db.Exec("INSERT INTO fact_vecs(fact_id, vec) VALUES(?,?)", a, store.VecToBlob(unitVec(5))); err != nil {
		t.Fatal(err)
	}
	bv := make([]float32, embedDims)
	bv[0] = 0.3
	if _, err := db.Exec("INSERT INTO fact_vecs(fact_id, vec) VALUES(?,?)", b, store.VecToBlob(bv)); err != nil {
		t.Fatal(err)
	}
	// a's paraphrase vec is a perfect match for the query
	addParaphrase(t, db, a, "alpha reworded", unitVec(0))

	hits, err := Search(db, "zzz nomatch lexical", "", 5, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].ID != a {
		t.Fatalf("cosine merge miss: hits=%+v want top=%d", hits, a)
	}
}

// TestParaArmExcludesSuperseded: a paraphrase of a dead fact must not surface it.
func TestParaArmExcludesSuperseded(t *testing.T) {
	db := store.TestDB(t)
	stubEmbedServer(t, func(string) []float32 { return make([]float32, embedDims) })

	now := float64(time.Now().Unix())
	old := store.InsertFact(t, db, "planner served from spot box", "fact", "quasar", now-7200)
	nw := store.InsertFact(t, db, "planner served from flex box", "fact", "quasar", now)
	store.Supersede(t, db, old, nw, now)
	addParaphrase(t, db, old, "unique wombat phrasing for the retired deployment", unitVec(1))

	hits, err := Search(db, "unique wombat phrasing retired deployment", "", 5, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.ID == old {
			t.Fatalf("superseded fact surfaced via paraphrase arm: %+v", hits)
		}
	}
}

// TestParaArmInertUnderAsOf: paraphrase coverage is live-only, so both para
// arms must be skipped in a historical frame — a paraphrase-only match may
// surface its fact today but not under --as-of, where the arm would favor
// whichever facts happened to survive to the present.
func TestParaArmInertUnderAsOf(t *testing.T) {
	db := store.TestDB(t)
	stubEmbedServer(t, func(string) []float32 { return make([]float32, embedDims) })

	now := float64(time.Now().Unix())
	live := store.InsertFact(t, db, "kubelet rotates serving certificates automatically", "fact", "infra", now-7200)
	addParaphrase(t, db, live, "TLS cert renewal happens on its own for worker daemons", nil)

	hits, err := Search(db, "TLS cert renewal worker daemons", "", 5, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if !containsFact(hits, live) {
		t.Fatalf("current frame: paraphrase arm should retrieve the fact, hits=%+v", hits)
	}
	// the fact was already valid an hour ago — but the para-only match must not
	// surface it in the historical frame
	hits, err = Search(db, "TLS cert renewal worker daemons", "", 5, now-3600, false)
	if err != nil {
		t.Fatal(err)
	}
	if containsFact(hits, live) {
		t.Fatalf("as-of frame: paraphrase arm must be inert, hits=%+v", hits)
	}
}

// TestParaphraseRunIsolatesPoisonStatement: a statement the model insists on
// splitting (shape error) must be skipped via paraphrase_skip — not wedge the
// backfill by being re-selected forever — while the rest of its batch lands.
func TestParaphraseRunIsolatesPoisonStatement(t *testing.T) {
	db := store.TestDB(t)
	stubEmbedServer(t, func(string) []float32 { return unitVec(0) })

	now := float64(time.Now().Unix())
	good1 := store.InsertFact(t, db, "alpha service listens on port 1234", "fact", "", now)
	poison := store.InsertFact(t, db, "batch had 131 of 220 tasks; 83 solved and 48 failed for 63%", "fact", "", now)
	good2 := store.InsertFact(t, db, "beta box uses systemd user units", "fact", "", now)

	hasPoison := func(stmts []string) bool {
		for _, s := range stmts {
			if strings.Contains(s, "131 of 220") {
				return true
			}
		}
		return false
	}
	batch := func(stmts []string) ([]string, error) {
		if hasPoison(stmts) { // model re-segments the compound statement
			return nil, fmt.Errorf("%w: count mismatch: got %d want %d", errParaShape, len(stmts)+1, len(stmts))
		}
		out := make([]string, len(stmts))
		for i, s := range stmts {
			out[i] = "reworded: " + s
		}
		return out, nil
	}

	facts, calls, err := paraphraseRunWith(db, 10, batch)
	if err != nil {
		t.Fatal(err)
	}
	if facts != 2 {
		t.Fatalf("facts=%d calls=%d, want 2 paraphrased", facts, calls)
	}
	for _, id := range []int64{good1, good2} {
		var n int
		if err := db.QueryRow("SELECT COUNT(*) FROM paraphrase_done WHERE fact_id=?", id).Scan(&n); err != nil || n != 1 {
			t.Fatalf("fact %d not marked done (n=%d err=%v)", id, n, err)
		}
	}
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM paraphrase_skip WHERE fact_id=?", poison).Scan(&n); err != nil || n != 1 {
		t.Fatalf("poison fact not skipped (n=%d err=%v)", n, err)
	}
	// a second run must find nothing left — the skip row prevents re-selection
	facts, calls, err = paraphraseRunWith(db, 10, batch)
	if err != nil || facts != 0 || calls != 0 {
		t.Fatalf("second run not idle: facts=%d calls=%d err=%v", facts, calls, err)
	}
}

// TestDecayedMassOrderSQL: the mass-DESC coverage query must run (sqlite exp())
// and rank a fresh heavy fact above an old light one.
func TestDecayedMassOrderSQL(t *testing.T) {
	db := store.TestDB(t)
	now := float64(time.Now().Unix())
	heavy := store.InsertFact(t, db, "heavy fresh fact", "fact", "", now)
	if _, err := db.Exec("UPDATE facts SET mass = 3.0 WHERE id = ?", heavy); err != nil {
		t.Fatal(err)
	}
	light := store.InsertFact(t, db, "old light status", "status", "", now-90*86400)
	rows, err := db.Query(`SELECT f.id FROM facts f
		LEFT JOIN paraphrase_done d ON d.fact_id = f.id
		WHERE f.superseded_at IS NULL AND d.fact_id IS NULL
		ORDER BY `+decayedMassOrder+` LIMIT 20`, now)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var order []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		order = append(order, id)
	}
	if len(order) != 2 || order[0] != heavy || order[1] != light {
		t.Fatalf("mass order wrong: %v (heavy=%d light=%d)", order, heavy, light)
	}
}
