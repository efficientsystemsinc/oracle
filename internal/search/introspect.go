package search

// kb ports, round 2: introspection snapshot for the reasoner, ask-trajectory
// memory (plan-cache-lite), and an optimize pass (orphans + merge suggestions).

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/store"
	"os"
	"sort"
	"strings"
	"time"
)

var introCache struct {
	text string
	ts   time.Time
}

// introspect summarizes what the graph contains so the reasoner plans with
// knowledge of coverage — kb's kbIntrospection, cached 5 min.
func Introspect(db *sql.DB) string {
	if time.Since(introCache.ts) < 5*time.Minute && introCache.text != "" {
		return introCache.text
	}
	var b strings.Builder
	var live, total int
	_ = db.QueryRow("SELECT COUNT(*) FROM facts WHERE superseded_at IS NULL").Scan(&live)
	_ = db.QueryRow("SELECT COUNT(*) FROM facts").Scan(&total)
	fmt.Fprintf(&b, "GRAPH CONTENTS: %d live facts (%d incl. history). ", live, total)
	part := func(q, label string) {
		rows, err := db.Query(q)
		if err != nil {
			return
		}
		defer rows.Close()
		var items []string
		for rows.Next() {
			var name string
			var c int
			_ = rows.Scan(&name, &c)
			items = append(items, fmt.Sprintf("%s(%d)", name, c))
		}
		if len(items) > 0 {
			fmt.Fprintf(&b, "%s: %s. ", label, strings.Join(items, ", "))
		}
	}
	part(`SELECT COALESCE(repo,'?'), COUNT(*) FROM facts WHERE superseded_at IS NULL GROUP BY repo ORDER BY 2 DESC LIMIT 12`, "Top repos")
	part(`SELECT COALESCE(etype,'untyped'), COUNT(*) FROM entities GROUP BY etype ORDER BY 2 DESC LIMIT 8`, "Entity types")
	part(`SELECT name, seen_count FROM predicates ORDER BY seen_count DESC LIMIT 12`, "Predicates")
	part(`SELECT m.name, COUNT(o.id) FROM metrics m JOIN metric_observations o ON o.metric_id=m.id GROUP BY m.id ORDER BY 2 DESC LIMIT 12`, "Metrics")
	introCache.text, introCache.ts = b.String(), time.Now()
	return introCache.text
}

// pastInvestigations returns similar prior ask trajectories (plan-cache-lite):
// what tool calls answered similar questions before.
func PastInvestigations(db *sql.DB, q string) string {
	rows, err := db.Query(`SELECT q, results FROM traces WHERE kind='ask'
		AND id IN (SELECT id FROM traces WHERE kind='ask' ORDER BY id DESC LIMIT 400)
		ORDER BY id DESC`)
	if err != nil {
		return ""
	}
	defer rows.Close()
	qTok := map[string]bool{}
	for _, t := range tokenRe.FindAllString(strings.ToLower(q), 12) {
		qTok[t] = true
	}
	best, bestScore := "", 2 // require >2 shared tokens
	for rows.Next() {
		var pq, res string
		_ = rows.Scan(&pq, &res)
		score := 0
		for _, t := range tokenRe.FindAllString(strings.ToLower(pq), 16) {
			if qTok[t] {
				score++
			}
		}
		if score > bestScore {
			bestScore, best = score, fmt.Sprintf("A similar past question %q was answered via: %s", pq, store.Truncate(res, 600))
		}
	}
	return best
}

// logAskTrace records the investigation (tool calls) and which facts the final
// answer cited — the citations are the relevance labels the self-eval and any
// future learned ranker train against.
func LogAskTrace(db *sql.DB, q string, calls []string, cited []int64) {
	payload, _ := json.Marshal(map[string]any{"calls": calls, "cited": cited})
	_, _ = db.Exec("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
		float64(time.Now().Unix()), "ask", q, string(payload))
}

// optimize: kb's kbOptimization — orphan cleanup + safe merge suggestions.
func Optimize(db *sql.DB, apply bool) (string, error) {
	var b strings.Builder
	del := func(q, label string) {
		res, err := db.Exec(q)
		if err != nil {
			fmt.Fprintf(&b, "%s: ERROR %v\n", label, err)
			return
		}
		n, _ := res.RowsAffected()
		fmt.Fprintf(&b, "%s: %d removed\n", label, n)
	}
	del(`DELETE FROM fact_entities WHERE fact_id NOT IN (SELECT id FROM facts)
		OR entity_id NOT IN (SELECT id FROM entities)`, "orphan fact_entities")
	del(`DELETE FROM triples WHERE fact_id NOT IN (SELECT id FROM facts)
		OR subject_id NOT IN (SELECT id FROM entities)
		OR (object_id IS NOT NULL AND object_id NOT IN (SELECT id FROM entities))`, "orphan triples")
	del(`DELETE FROM metric_observations WHERE entity_id IS NOT NULL AND entity_id NOT IN (SELECT id FROM entities)`, "orphan observations")
	del(`DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM fact_entities)
		AND id NOT IN (SELECT subject_id FROM triples)
		AND id NOT IN (SELECT object_id FROM triples WHERE object_id IS NOT NULL)
		AND id NOT IN (SELECT entity_id FROM metric_observations WHERE entity_id IS NOT NULL)`, "unreferenced entities")

	// junk repo labels (scratch dirs named after prompts/URLs) -> unknown;
	// same predicate ingest applies via repoFromCwd, retrofit for old rows
	for _, tbl := range []string{"facts", "files"} {
		rows, err := db.Query(`SELECT DISTINCT repo FROM ` + tbl + ` WHERE repo IS NOT NULL`)
		if err != nil {
			return b.String(), err
		}
		var bad []string
		for rows.Next() {
			var r string
			_ = rows.Scan(&r)
			if !ingest.ValidRepoName(r) {
				bad = append(bad, r)
			}
		}
		rows.Close()
		for _, r := range bad {
			res, err := db.Exec(`UPDATE `+tbl+` SET repo = 'unknown' WHERE repo = ?`, r)
			if err != nil {
				return b.String(), err
			}
			n, _ := res.RowsAffected()
			fmt.Fprintf(&b, "junk repo %q -> unknown (%d %s rows)\n", r, n, tbl)
		}
	}

	// merge suggestions: trivial name variants, grouped by the same vkey that
	// link-time folding uses — leftovers here predate folding or were ambiguous
	rows, err := db.Query("SELECT name, seen_count FROM entities")
	if err != nil {
		return b.String(), err
	}
	type ent struct {
		name string
		seen int
	}
	byNorm := map[string][]ent{}
	for rows.Next() {
		var e ent
		_ = rows.Scan(&e.name, &e.seen)
		byNorm[kb.VariantKey(e.name)] = append(byNorm[kb.VariantKey(e.name)], e)
	}
	rows.Close()
	merged := 0
	for _, group := range byNorm {
		if len(group) < 2 {
			continue
		}
		// winner = most seen
		w := group[0]
		for _, e := range group[1:] {
			if e.seen > w.seen {
				w = e
			}
		}
		for _, e := range group {
			if e.name == w.name {
				continue
			}
			if apply {
				if err := kb.MergeEntities(db, w.name, e.name); err == nil {
					merged++
				}
			} else {
				fmt.Fprintf(&b, "suggest merge: %q -> %q\n", e.name, w.name)
			}
		}
	}
	if apply {
		fmt.Fprintf(&b, "variant merges applied: %d\n", merged)
	}
	return b.String(), nil
}

// sessionCwd scans a session jsonl for its first recorded cwd (claude records
// carry cwd per line; codex in session_meta). Capped so huge transcripts don't
// stall the relabel pass.
func sessionCwd(path, source string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	parse := ingest.ParseClaudeLine
	if source == "codex" {
		parse = ingest.ParseCodexLine
	}
	rd := bufio.NewReaderSize(f, 1<<20)
	for lines, read := 0, 0; lines < 2000 && read < 32<<20; lines++ {
		line, err := rd.ReadBytes('\n')
		if len(line) > 0 {
			read += len(line)
			if p := parse(line); p.Cwd != "" {
				return p.Cwd
			}
		}
		if err != nil {
			break
		}
	}
	return ""
}

// relabelRepos re-derives each tracked file's repo label from its session cwd
// via the git-aware resolver and migrates files + facts rows whose current
// label maps to a resolvable better one. Dry-run prints the migration plan;
// apply performs the UPDATEs. Facts only move when they still carry the file's
// old label, so per-fact repos from mid-session cwd changes are left alone.
func RelabelRepos(db *sql.DB, apply bool) (string, error) {
	rows, err := db.Query(`SELECT path, source, COALESCE(repo,'') FROM files`)
	if err != nil {
		return "", err
	}
	type frow struct{ path, source, repo string }
	var files []frow
	for rows.Next() {
		var f frow
		if err := rows.Scan(&f.path, &f.source, &f.repo); err != nil {
			rows.Close()
			return "", err
		}
		files = append(files, f)
	}
	rows.Close()

	type mig struct{ files, facts int64 }
	plan := map[string]*mig{} // "old -> new"
	filesTouched, factsTouched := int64(0), int64(0)
	for _, f := range files {
		cwd := sessionCwd(f.path, f.source)
		if cwd == "" {
			continue
		}
		newLabel := ingest.RepoFromCwd(cwd)
		if newLabel == "unknown" || newLabel == f.repo {
			continue
		}
		var nfacts int64
		if err := db.QueryRow(`SELECT COUNT(*) FROM facts WHERE src_path = ? AND COALESCE(repo,'') = ?`,
			f.path, f.repo).Scan(&nfacts); err != nil {
			return "", err
		}
		key := fmt.Sprintf("%q -> %q", f.repo, newLabel)
		if plan[key] == nil {
			plan[key] = &mig{}
		}
		plan[key].files++
		plan[key].facts += nfacts
		filesTouched++
		factsTouched += nfacts
		if apply {
			if _, err := db.Exec(`UPDATE facts SET repo = ? WHERE src_path = ? AND COALESCE(repo,'') = ?`,
				newLabel, f.path, f.repo); err != nil {
				return "", err
			}
			if _, err := db.Exec(`UPDATE files SET repo = ? WHERE path = ?`, newLabel, f.path); err != nil {
				return "", err
			}
		}
	}

	var b strings.Builder
	mode := "DRY-RUN (pass --apply to relabel)"
	if apply {
		mode = "APPLIED"
	}
	fmt.Fprintf(&b, "relabel %s: %d files, %d facts across %d label migrations\n",
		mode, filesTouched, factsTouched, len(plan))
	keys := make([]string, 0, len(plan))
	for k := range plan {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return plan[keys[i]].facts > plan[keys[j]].facts })
	for _, k := range keys {
		fmt.Fprintf(&b, "  %s: %d facts, %d files\n", k, plan[k].facts, plan[k].files)
	}
	return b.String(), nil
}
