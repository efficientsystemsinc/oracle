package kb

// Full port of the remaining nebulatgs/kb surface: typed entities, bitemporal
// entity aliases, entity merge, S-P-O triples + predicate registry, metric
// observations, n-hop graph traversal, topic clustering, narrative builder.

import (
	"database/sql"
	"fmt"
	"oracle/internal/llm"
	"oracle/internal/store"
	"sort"
	"strings"
	"time"
)

// variantKey collapses trivial name variants (case, hyphen, underscore, dot,
// space) to one key: "upload_delta", "Upload-Delta", "upload delta" -> "uploaddelta".
// The vkey backfill in applyKBSchema must compute the same thing in SQL.
func VariantKey(name string) string {
	return strings.NewReplacer("-", "", "_", "", ".", "", " ", "").Replace(strings.ToLower(name))
}

// ---------- alias resolution ----------

// resolveEntityExact: name -> entity id via canonical name, then live alias.
// The well-defined part of resolution — correctness-sensitive paths (e.g.
// supersession candidate scoping) must use this, never the prefix fallback.
func ResolveEntityExact(db store.DBQ, name string) (int64, string, bool) {
	canon := strings.ToLower(strings.TrimSpace(name))
	var id int64
	var display string
	if db.QueryRow("SELECT id, display FROM entities WHERE name = ?", canon).Scan(&id, &display) == nil {
		return id, display, true
	}
	if db.QueryRow(`SELECT e.id, e.display FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
		WHERE a.alias = ? AND a.valid_to IS NULL`, canon).Scan(&id, &display) == nil {
		return id, display, true
	}
	return 0, "", false
}

// resolveEntity adds a prefix-match fallback on top of exact resolution, for
// human input: CLI, HTTP paths, and ask tool arguments.
func ResolveEntity(db store.DBQ, name string) (int64, string, bool) {
	if id, display, ok := ResolveEntityExact(db, name); ok {
		return id, display, true
	}
	canon := strings.ToLower(strings.TrimSpace(name))
	var id int64
	var display string
	if db.QueryRow("SELECT id, display FROM entities WHERE name LIKE ? ORDER BY seen_count DESC",
		canon+"%").Scan(&id, &display) == nil {
		return id, display, true
	}
	return 0, "", false
}

func AddAlias(db store.DBQ, entityID int64, alias string, conf float64) error {
	canon := strings.ToLower(strings.TrimSpace(alias))
	if canon == "" {
		return fmt.Errorf("empty alias")
	}
	var exists int64
	if db.QueryRow(`SELECT id FROM entity_aliases WHERE alias = ? AND entity_id = ? AND valid_to IS NULL`,
		canon, entityID).Scan(&exists) == nil {
		return nil
	}
	now := float64(time.Now().Unix())
	_, err := db.Exec(`INSERT INTO entity_aliases(entity_id, alias, valid_from, recorded_at, confidence)
		VALUES(?,?,?,?,?)`, entityID, canon, now, now, conf)
	return err
}

// mergeEntities folds loser into winner: facts, edges, aliases, triples, observations.
// The loser's name survives as a live alias. kb listed this as its own unbuilt gap.
func MergeEntities(db *sql.DB, winner, loser string) error {
	wid, wdisp, ok := ResolveEntity(db, winner)
	if !ok {
		return fmt.Errorf("winner %q not found", winner)
	}
	lid, ldisp, ok := ResolveEntity(db, loser)
	if !ok {
		return fmt.Errorf("loser %q not found", loser)
	}
	if wid == lid {
		return fmt.Errorf("%q and %q already resolve to the same entity", winner, loser)
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	now := float64(time.Now().Unix())
	if _, err := tx.Exec("UPDATE OR IGNORE fact_entities SET entity_id = ? WHERE entity_id = ?", wid, lid); err != nil {
		return err
	}
	if _, err := tx.Exec("DELETE FROM fact_entities WHERE entity_id = ?", lid); err != nil {
		return err
	}
	for _, col := range []string{"subject_id", "object_id"} {
		if _, err := tx.Exec("UPDATE triples SET "+col+" = ? WHERE "+col+" = ?", wid, lid); err != nil {
			return err
		}
	}
	if _, err := tx.Exec("UPDATE metric_observations SET entity_id = ? WHERE entity_id = ?", wid, lid); err != nil {
		return err
	}
	if _, err := tx.Exec("UPDATE entity_aliases SET entity_id = ? WHERE entity_id = ?", wid, lid); err != nil {
		return err
	}
	// co-mention edges: repoint, folding duplicates
	rows, err := tx.Query("SELECT a, b, count FROM entity_edges WHERE a = ? OR b = ?", lid, lid)
	if err != nil {
		return err
	}
	type edge struct{ a, b, c int64 }
	var edges []edge
	for rows.Next() {
		var e edge
		_ = rows.Scan(&e.a, &e.b, &e.c)
		edges = append(edges, e)
	}
	rows.Close()
	if _, err := tx.Exec("DELETE FROM entity_edges WHERE a = ? OR b = ?", lid, lid); err != nil {
		return err
	}
	for _, e := range edges {
		other := e.a
		if other == lid {
			other = e.b
		}
		if other == wid {
			continue
		}
		lo, hi := min(wid, other), max(wid, other)
		if _, err := tx.Exec(`INSERT INTO entity_edges(a,b,count,last_seen) VALUES(?,?,?,?)
			ON CONFLICT(a,b) DO UPDATE SET count = count + excluded.count`, lo, hi, e.c, now); err != nil {
			return err
		}
	}
	// loser's names become live aliases of the winner
	var lname string
	_ = tx.QueryRow("SELECT name FROM entities WHERE id = ?", lid).Scan(&lname)
	if _, err := tx.Exec(`INSERT INTO entity_aliases(entity_id, alias, valid_from, recorded_at, confidence)
		VALUES(?,?,?,?,1.0)`, wid, lname, now, now); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE entities SET seen_count = seen_count +
		(SELECT seen_count FROM entities WHERE id = ?) WHERE id = ?`, lid, wid); err != nil {
		return err
	}
	if _, err := tx.Exec("DELETE FROM entities WHERE id = ?", lid); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	fmt.Printf("merged %q -> %q (alias kept)\n", ldisp, wdisp)
	return nil
}

// ---------- triples + metrics upsert (used by ingest + enrich) ----------

type Triple struct {
	Subject   string `json:"subject"`
	Predicate string `json:"predicate"`
	Object    string `json:"object"`
	Literal   bool   `json:"literal"`
}

type Observation struct {
	Metric  string  `json:"metric"`
	Entity  string  `json:"entity"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	DateISO string  `json:"date"`
}

// validEntityName rejects path fragments, sentences, and other junk the LLM
// occasionally emits as "entities".
func ValidEntityName(canon string) bool {
	if canon == "" || len(canon) > 48 {
		return false
	}
	if strings.ContainsAny(canon, "/\\`\"«»") {
		return false
	}
	return strings.Count(canon, " ") <= 3
}

// resolveOrCreateEntity is the single write path for entity mentions:
// exact name -> live alias -> unique trivial-variant match (recording the
// observed spelling as an alias so it resolves exactly next time) -> create.
// Folding at link time is what makes `oracle merge` stick: without the alias
// step, the next ingest mentioning a merged-away name would resurrect it.
// An ambiguous variant (two existing entities share the key) creates a new
// entity instead of guessing; optimize surfaces those for explicit merging.
func ResolveOrCreateEntity(q store.DBQ, name, etype string, now float64) (int64, error) {
	canon := strings.ToLower(strings.TrimSpace(name))
	if !ValidEntityName(canon) {
		return 0, fmt.Errorf("junk entity name %q", canon)
	}
	id, _, ok := ResolveEntityExact(q, canon)
	if !ok {
		rows, err := q.Query("SELECT id FROM entities WHERE vkey = ? LIMIT 2", VariantKey(canon))
		if err != nil {
			return 0, err
		}
		var matches []int64
		for rows.Next() {
			var m int64
			_ = rows.Scan(&m)
			matches = append(matches, m)
		}
		rows.Close()
		if len(matches) == 1 {
			id, ok = matches[0], true
			if err := AddAlias(q, id, canon, 1.0); err != nil {
				return 0, err
			}
		}
	}
	if ok {
		if _, err := q.Exec("UPDATE entities SET seen_count = seen_count + 1, last_seen = ? WHERE id = ?", now, id); err != nil {
			return 0, err
		}
		if etype != "" {
			_, _ = q.Exec("UPDATE entities SET etype = COALESCE(etype, ?) WHERE id = ?", etype, id)
		}
		return id, nil
	}
	res, err := q.Exec(`INSERT INTO entities(name, display, etype, seen_count, last_seen, vkey)
		VALUES(?,?,?,1,?,?)`, canon, strings.TrimSpace(name), nullify(etype), now, VariantKey(canon))
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

func UpsertEntityByName(db store.DBQ, name, etype string, now float64) (int64, error) {
	return ResolveOrCreateEntity(db, name, etype, now)
}

func nullify(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func StoreTriples(db *sql.DB, factID int64, trips []Triple, now float64) error {
	for _, t := range trips {
		if t.Subject == "" || t.Predicate == "" || t.Object == "" {
			continue
		}
		sid, err := UpsertEntityByName(db, t.Subject, "", now)
		if err != nil {
			continue // junk subject: drop this triple
		}
		pred := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(t.Predicate), " ", "_"))
		// canonicalization: a predicate folded away by `oracle canonpreds` must
		// not be resurrected — route the mention to its canonical predicate.
		var pid int64
		if err := db.QueryRow("SELECT predicate_id FROM predicate_aliases WHERE alias = ?", pred).Scan(&pid); err == nil {
			if _, err := db.Exec("UPDATE predicates SET seen_count = seen_count + 1 WHERE id = ?", pid); err != nil {
				return err
			}
		} else if err != sql.ErrNoRows {
			return err
		} else {
			if _, err := db.Exec(`INSERT INTO predicates(name, seen_count) VALUES(?,1)
				ON CONFLICT(name) DO UPDATE SET seen_count = seen_count + 1`, pred); err != nil {
				return err
			}
			if err := db.QueryRow("SELECT id FROM predicates WHERE name = ?", pred).Scan(&pid); err != nil {
				return err
			}
		}
		if t.Literal {
			if _, err := db.Exec(`INSERT INTO triples(fact_id, subject_id, predicate_id, object_literal, recorded_at)
				VALUES(?,?,?,?,?)`, factID, sid, pid, t.Object, now); err != nil {
				return err
			}
		} else {
			oid, err := UpsertEntityByName(db, t.Object, "", now)
			if err != nil {
				continue // junk object: drop this triple
			}
			if _, err := db.Exec(`INSERT INTO triples(fact_id, subject_id, predicate_id, object_id, recorded_at)
				VALUES(?,?,?,?,?)`, factID, sid, pid, oid, now); err != nil {
				return err
			}
		}
	}
	return nil
}

func StoreObservations(db *sql.DB, factID int64, obs []Observation, factValidFrom, now float64) error {
	for _, o := range obs {
		if o.Metric == "" {
			continue
		}
		mname := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(o.Metric), " ", "_"))
		if _, err := db.Exec(`INSERT INTO metrics(name, unit, seen_count) VALUES(?,?,1)
			ON CONFLICT(name) DO UPDATE SET seen_count = seen_count + 1, unit = COALESCE(metrics.unit, excluded.unit)`,
			mname, nullify(o.Unit)); err != nil {
			return err
		}
		var mid int64
		if err := db.QueryRow("SELECT id FROM metrics WHERE name = ?", mname).Scan(&mid); err != nil {
			return err
		}
		var eid any
		if o.Entity != "" {
			if id, err := UpsertEntityByName(db, o.Entity, "", now); err == nil {
				eid = id
			}
		}
		occurred := factValidFrom
		if o.DateISO != "" {
			if t, err := time.Parse("2006-01-02", o.DateISO); err == nil {
				occurred = float64(t.Unix())
			}
		}
		if _, err := db.Exec(`INSERT INTO metric_observations(metric_id, entity_id, value, occurred_at, fact_id, recorded_at)
			VALUES(?,?,?,?,?,?)`, mid, eid, o.Value, occurred, factID, now); err != nil {
			return err
		}
	}
	return nil
}

// ---------- graph traversal ----------

func Traverse(db *sql.DB, start string, hops int, limit int) (map[string]any, error) {
	sid, disp, ok := ResolveEntity(db, start)
	if !ok {
		return map[string]any{"entity": nil}, nil
	}
	type hop struct {
		id   int64
		dist int
	}
	seen := map[int64]int{sid: 0}
	frontier := []hop{{sid, 0}}
	var links []map[string]any
	for len(frontier) > 0 {
		cur := frontier[0]
		frontier = frontier[1:]
		if cur.dist >= hops {
			continue
		}
		// one neighbor stream: typed triples (out+in) first, then co-mentions
		// by strength. Co-mentions keep unenriched neighbourhoods reachable —
		// only enriched facts have triples — but a co-mention row to a node
		// already reached is skipped: the typed link is strictly more informative.
		rows, err := db.Query(`SELECT pred, nid, disp, dir, cnt FROM (
			SELECT p.name AS pred, e2.id AS nid, e2.display AS disp, 'out' AS dir, NULL AS cnt, 0 AS pri
			  FROM triples t JOIN predicates p ON p.id = t.predicate_id JOIN entities e2 ON e2.id = t.object_id
			  WHERE t.subject_id = ?
			UNION ALL
			SELECT p.name, e1.id, e1.display, 'in', NULL, 0
			  FROM triples t JOIN predicates p ON p.id = t.predicate_id JOIN entities e1 ON e1.id = t.subject_id
			  WHERE t.object_id = ?
			UNION ALL
			SELECT 'co_mentioned', e.id, e.display, 'co', ee.count, 1
			  FROM entity_edges ee JOIN entities e ON e.id = CASE WHEN ee.a = ? THEN ee.b ELSE ee.a END
			  WHERE ee.a = ? OR ee.b = ?
			) ORDER BY pri, cnt DESC LIMIT ?`, cur.id, cur.id, cur.id, cur.id, cur.id, limit)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var pred, disp2, dir string
			var nid int64
			var cnt sql.NullInt64
			_ = rows.Scan(&pred, &nid, &disp2, &dir, &cnt)
			_, dup := seen[nid]
			if dir == "co" && dup {
				continue
			}
			if !dup {
				seen[nid] = cur.dist + 1
				frontier = append(frontier, hop{nid, cur.dist + 1})
			}
			link := map[string]any{"from": cur.id, "predicate": pred,
				"to": disp2, "dir": dir, "depth": cur.dist + 1}
			if cnt.Valid {
				link["count"] = int(cnt.Int64)
			}
			links = append(links, link)
			if len(links) >= limit {
				break
			}
		}
		rows.Close()
		if len(links) >= limit {
			break
		}
	}
	return map[string]any{"entity": disp, "hops": hops, "links": links, "reached": len(seen) - 1}, nil
}

// ---------- topics (embedding k-means + LLM labels) ----------

func Topics(db *sql.DB, k int) ([]map[string]any, error) {
	rows, err := db.Query(`SELECT f.id, f.statement, v.vec FROM facts f
		JOIN fact_vecs v ON v.fact_id = f.id WHERE f.superseded_at IS NULL`)
	if err != nil {
		return nil, err
	}
	var ids []int64
	var stmts []string
	var vecs [][]float32
	for rows.Next() {
		var id int64
		var s string
		var blob []byte
		if err := rows.Scan(&id, &s, &blob); err != nil {
			return nil, err
		}
		ids = append(ids, id)
		stmts = append(stmts, s)
		vecs = append(vecs, store.BlobToVec(blob))
	}
	rows.Close()
	if len(vecs) < k {
		return nil, fmt.Errorf("only %d facts, need >= k=%d", len(vecs), k)
	}
	assign := kmeans(vecs, k, 12)
	out := make([]map[string]any, 0, k)
	for c := 0; c < k; c++ {
		var sample []string
		n := 0
		for i, a := range assign {
			if a == c {
				n++
				if len(sample) < 12 {
					sample = append(sample, stmts[i])
				}
			}
		}
		if n == 0 {
			continue
		}
		var lab struct {
			Label string `json:"label"`
		}
		if err := llm.ChatJSON(`Name this cluster of engineering facts with a 2-5 word topic label. Return JSON {"label": str}`,
			strings.Join(sample, "\n"), 500, &lab); err != nil {
			return nil, err
		}
		out = append(out, map[string]any{"label": lab.Label, "size": n, "sample": sample[:min(3, len(sample))]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i]["size"].(int) > out[j]["size"].(int) })
	return out, nil
}

func kmeans(vecs [][]float32, k, iters int) []int {
	n, d := len(vecs), len(vecs[0])
	cents := make([][]float64, k)
	for c := 0; c < k; c++ {
		cents[c] = make([]float64, d)
		for j, x := range vecs[(c*n)/k] {
			cents[c][j] = float64(x)
		}
	}
	assign := make([]int, n)
	for it := 0; it < iters; it++ {
		for i, v := range vecs {
			best, bs := 0, -2.0
			for c := 0; c < k; c++ {
				var s float64
				for j, x := range v {
					s += float64(x) * cents[c][j]
				}
				if s > bs {
					bs, best = s, c
				}
			}
			assign[i] = best
		}
		for c := 0; c < k; c++ {
			cnt := 0
			nc := make([]float64, d)
			for i, a := range assign {
				if a == c {
					cnt++
					for j, x := range vecs[i] {
						nc[j] += float64(x)
					}
				}
			}
			if cnt > 0 {
				for j := range nc {
					nc[j] /= float64(cnt)
				}
				cents[c] = nc
			}
		}
	}
	return assign
}

// ---------- narrative builder ----------

func Narrative(db *sql.DB, subject string) (string, error) {
	// gather everything about the subject (entity or repo), time-ordered, incl. superseded history
	eid, disp, isEnt := ResolveEntity(db, subject)
	var rows *sql.Rows
	var err error
	if isEnt {
		rows, err = db.Query(`SELECT f.statement, f.kind, f.valid_from, f.superseded_at IS NOT NULL
			FROM fact_entities fe JOIN facts f ON f.id = fe.fact_id
			WHERE fe.entity_id = ? ORDER BY f.valid_from LIMIT 250`, eid)
	} else {
		disp = subject
		rows, err = db.Query(`SELECT statement, kind, valid_from, superseded_at IS NOT NULL
			FROM facts WHERE repo = ? ORDER BY valid_from LIMIT 250`, subject)
	}
	if err != nil {
		return "", err
	}
	defer rows.Close()
	var b strings.Builder
	nRows := 0
	for rows.Next() {
		var s, kind string
		var vf float64
		var dead bool
		_ = rows.Scan(&s, &kind, &vf, &dead)
		tag := ""
		if dead {
			tag = " [superseded]"
		}
		fmt.Fprintf(&b, "%s [%s]%s %s\n", store.LocalDate(vf), kind, tag, s)
		nRows++
	}
	if nRows == 0 {
		return "", fmt.Errorf("nothing known about %q", subject)
	}
	var out struct {
		Narrative string `json:"narrative"`
	}
	if err := llm.ChatJSON(`Write a tight chronological narrative (3-8 short paragraphs) of this subject's history from the dated facts. Superseded facts are past states — narrate the transitions. No invention beyond the facts. Return JSON {"narrative": str}`,
		"SUBJECT: "+disp+"\n\n"+b.String(), 8000, &out); err != nil {
		return "", err
	}
	return out.Narrative, nil
}
