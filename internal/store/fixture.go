package store

// `oracle fixture --out PATH` — deterministic regression fixture snapshot.
//
// Builds a small (~2000-fact) standalone oracle.db from the current DB:
//   - the most-corroborated + most-recent LIVE facts per repo,
//   - 100 complete supersession chains (every link, head to tail),
//   - the newest cited `ask` traces plus their cited facts,
//   - all entities / aliases / fact_entities / entity_edges / fact_vecs /
//     edges rows referenced by the selected facts.
//
// VACUUM-INTO semantics: the fixture is assembled in a temp file inside a
// single transaction, then atomically renamed over --out. The source DB is
// attached read-only and never written.

import (
	"database/sql"
	"fmt"
	"os"
	"strings"

	_ "modernc.org/sqlite"
)

const fixtureFactBudget = 1800 // per-repo selection budget; chains + cited facts land on top

// openFixtureDB creates a fresh, empty oracle-schema DB at path (rollback
// journal, single file — no WAL sidecars in the committed fixture).
func OpenFixtureDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", fmt.Sprintf("file:%s?_pragma=journal_mode(DELETE)", path))
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("fixture schema: %w", err)
	}
	if _, err := db.Exec(embedSchema); err != nil {
		return nil, fmt.Errorf("fixture embed schema: %w", err)
	}
	if err := applyKBSchema(db); err != nil {
		return nil, err
	}
	if err := applyParaSchema(db); err != nil {
		return nil, err
	}
	return db, nil
}

// chainMembers walks a supersession chain in both directions from seed and
// returns every fact id on it (ancestors via superseded_by = X, descendants
// via following superseded_by). Bounded to keep pathological graphs finite.
func chainMembers(db DBQ, seed int64) ([]int64, error) {
	seen := map[int64]bool{seed: true}
	frontier := []int64{seed}
	for len(frontier) > 0 && len(seen) < 200 {
		cur := frontier[0]
		frontier = frontier[1:]
		// descendant: what superseded cur
		var next sql.NullInt64
		err := db.QueryRow("SELECT superseded_by FROM facts WHERE id = ?", cur).Scan(&next)
		if err != nil && err != sql.ErrNoRows {
			return nil, err
		}
		if next.Valid && next.Int64 != 0 && !seen[next.Int64] {
			seen[next.Int64] = true
			frontier = append(frontier, next.Int64)
		}
		// ancestors: everything cur superseded (chains can merge: N olds -> 1 new)
		rows, err := db.Query("SELECT id FROM facts WHERE superseded_by = ?", cur)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id int64
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			if !seen[id] {
				seen[id] = true
				frontier = append(frontier, id)
			}
		}
		rows.Close()
	}
	out := make([]int64, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	return out, nil
}

func BuildFixture(srcPath, outPath string) error {
	if _, err := os.Stat(srcPath); err != nil {
		return fmt.Errorf("source db: %w", err)
	}
	tmp := outPath + ".tmp"
	_ = os.Remove(tmp)
	dst, err := OpenFixtureDB(tmp)
	if err != nil {
		return err
	}
	defer dst.Close()
	defer os.Remove(tmp) // no-op after successful rename

	if _, err := dst.Exec("ATTACH ? AS src", "file:"+srcPath+"?mode=ro"); err != nil {
		return fmt.Errorf("attach source: %w", err)
	}

	tx, err := dst.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec("CREATE TEMP TABLE sel(id INTEGER PRIMARY KEY)"); err != nil {
		return err
	}

	// 1) per-repo live facts: most corroborated first, then most recent.
	// Deterministic: every ORDER BY tie-breaks on id.
	var repos int
	if err := tx.QueryRow(`SELECT COUNT(DISTINCT COALESCE(repo,'')) FROM src.facts
		WHERE superseded_at IS NULL`).Scan(&repos); err != nil {
		return err
	}
	if repos == 0 {
		return fmt.Errorf("source db has no live facts — refusing to build an empty fixture")
	}
	perRepo := fixtureFactBudget / repos
	if perRepo < 2 {
		perRepo = 2 // long-tail repos still contribute their top 2
	}
	if _, err := tx.Exec(`INSERT OR IGNORE INTO sel
		SELECT id FROM (
		  SELECT id, ROW_NUMBER() OVER (
		    PARTITION BY COALESCE(repo,'')
		    ORDER BY corroborations DESC, recorded_at DESC, id DESC) AS rn
		  FROM src.facts WHERE superseded_at IS NULL)
		WHERE rn <= ?`, perRepo); err != nil {
		return fmt.Errorf("per-repo selection: %w", err)
	}

	// 2) 100 complete supersession chains, newest first (seeded from the
	// newest superseded facts whose old/new are separated in world time, so
	// the fixture keeps feeding selfeval's era replay).
	seedRows, err := tx.Query(`SELECT f.id FROM src.facts f JOIN src.facts s ON s.id = f.superseded_by
		WHERE s.valid_from > f.valid_from + 3600 ORDER BY f.id DESC LIMIT 400`)
	if err != nil {
		return err
	}
	var seeds []int64
	for seedRows.Next() {
		var id int64
		if err := seedRows.Scan(&id); err != nil {
			seedRows.Close()
			return err
		}
		seeds = append(seeds, id)
	}
	seedRows.Close()
	chainDone := map[int64]bool{} // fact id -> already part of a captured chain
	chains := 0
	for _, seed := range seeds {
		if chains >= 100 {
			break
		}
		if chainDone[seed] {
			continue
		}
		members, err := chainMembers(&srcQ{tx}, seed)
		if err != nil {
			return err
		}
		for _, id := range members {
			chainDone[id] = true
			if _, err := tx.Exec("INSERT OR IGNORE INTO sel(id) VALUES(?)", id); err != nil {
				return err
			}
		}
		chains++
	}

	// 3) newest cited ask traces + their cited facts (citation replay).
	traceRows, err := tx.Query(`SELECT id, ts, kind, q, results FROM src.traces
		WHERE kind='ask' AND results LIKE '%"cited"%' ORDER BY id DESC LIMIT 200`)
	if err != nil {
		return err
	}
	type traceRow struct {
		id     int64
		ts     float64
		kind   string
		q, res sql.NullString
	}
	var trs []traceRow
	for traceRows.Next() {
		var t traceRow
		if err := traceRows.Scan(&t.id, &t.ts, &t.kind, &t.q, &t.res); err != nil {
			traceRows.Close()
			return err
		}
		trs = append(trs, t)
	}
	traceRows.Close()
	for _, t := range trs {
		if _, err := tx.Exec(`INSERT INTO traces(id, ts, kind, q, results) VALUES(?,?,?,?,?)`,
			t.id, t.ts, t.kind, t.q, t.res); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(`INSERT OR IGNORE INTO sel
		SELECT j.value FROM traces t, json_each(t.results, '$.cited') j
		WHERE EXISTS (SELECT 1 FROM src.facts f WHERE f.id = j.value)`); err != nil {
		return fmt.Errorf("cited fact selection: %w", err)
	}

	// 4) copy the selected facts + everything they reference.
	// facts_fts fills itself via the insert trigger in the fixture schema.
	copies := []struct{ label, stmt string }{
		{"facts", `INSERT INTO facts SELECT f.* FROM src.facts f JOIN sel ON sel.id = f.id`},
		{"fact_entities", `INSERT INTO fact_entities SELECT fe.* FROM src.fact_entities fe JOIN sel ON sel.id = fe.fact_id`},
		{"entities", `INSERT INTO entities SELECT e.* FROM src.entities e
			WHERE e.id IN (SELECT fe.entity_id FROM src.fact_entities fe JOIN sel ON sel.id = fe.fact_id)`},
		{"entity_aliases", `INSERT INTO entity_aliases SELECT a.* FROM src.entity_aliases a
			WHERE a.entity_id IN (SELECT id FROM entities)`},
		{"entity_edges", `INSERT INTO entity_edges SELECT ee.* FROM src.entity_edges ee
			WHERE ee.a IN (SELECT id FROM entities) AND ee.b IN (SELECT id FROM entities)`},
		{"fact_vecs", `INSERT INTO fact_vecs SELECT v.* FROM src.fact_vecs v JOIN sel ON sel.id = v.fact_id`},
		{"edges", `INSERT INTO edges SELECT e.* FROM src.edges e
			JOIN sel s1 ON s1.id = e.src JOIN sel s2 ON s2.id = e.dst`},
	}
	counts := map[string]int64{}
	for _, c := range copies {
		res, err := tx.Exec(c.stmt)
		if err != nil {
			return fmt.Errorf("copy %s: %w", c.label, err)
		}
		counts[c.label], _ = res.RowsAffected()
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	if _, err := dst.Exec("DETACH src"); err != nil {
		return err
	}
	if err := dst.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, outPath); err != nil {
		return err
	}
	fmt.Printf("fixture: %s\n", outPath)
	fmt.Printf("  repos: %d (per-repo cap %d)\n", repos, perRepo)
	fmt.Printf("  supersession chains: %d\n", chains)
	fmt.Printf("  ask traces: %d\n", len(trs))
	for _, k := range []string{"facts", "entities", "entity_aliases", "fact_entities", "entity_edges", "fact_vecs", "edges"} {
		fmt.Printf("  %s: %d\n", k, counts[k])
	}
	return nil
}

// srcQ adapts *sql.Tx to dbq with src.facts as the facts table for chain walks.
type srcQ struct{ tx *sql.Tx }

func (s *srcQ) Exec(q string, args ...any) (sql.Result, error) {
	return s.tx.Exec(rewriteSrc(q), args...)
}
func (s *srcQ) Query(q string, args ...any) (*sql.Rows, error) {
	return s.tx.Query(rewriteSrc(q), args...)
}
func (s *srcQ) QueryRow(q string, args ...any) *sql.Row {
	return s.tx.QueryRow(rewriteSrc(q), args...)
}
func rewriteSrc(q string) string { return strings.ReplaceAll(q, "FROM facts", "FROM src.facts") }
