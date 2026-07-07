package search

// Graph ops: ingest cycle (watch -> extract -> upsert w/ supersede),
// decay-weighted search, briefs, entity views, as-of queries.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"oracle/internal/ingest"
	"oracle/internal/kb"
	"oracle/internal/store"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// half-life (days) per kind — status/todo rot fast, preferences barely
var halfLifeDays = map[string]float64{
	"status": 7, "todo": 14, "fact": 60, "decision": 120, "gotcha": 120, "preference": 365,
}

const massEps = 0.05

type FactRow struct {
	ID         int64
	Statement  string
	Kind       string
	Repo       string
	Entities   string
	Files      string
	Confidence float64
	Mass       float64
	RecordedAt float64
	ValidFrom  float64
}

type FactOut struct {
	ID         int64    `json:"id"`
	Statement  string   `json:"statement"`
	Kind       string   `json:"kind"`
	Repo       string   `json:"repo"`
	Entities   []string `json:"entities"`
	Files      []string `json:"files"`
	Confidence float64  `json:"confidence"`
	Score      float64  `json:"score"`
	Mass       float64  `json:"mass"`
	AgeDays    float64  `json:"age_days"`
	Src        string   `json:"src"`
	Evidence   string   `json:"evidence,omitempty"`
	Corrob     int      `json:"corroborations,omitempty"`
	ContraBy   []int64  `json:"contradicted_by,omitempty"`
	ViaHistory bool     `json:"via_history,omitempty"`
	Stale      bool     `json:"stale"`
	AsOfDate   string   `json:"as_of_date"`
}

func massNow(kind string, mass, validFrom, now float64) float64 {
	hl := halfLifeDays[kind]
	if hl == 0 {
		hl = 60
	}
	age := math.Max(0, now-validFrom)
	return massEps + mass*math.Exp(-0.693*age/(hl*86400))
}

var tokenRe = regexp.MustCompile(`[A-Za-z0-9_.-]{2,}`)

// ftsStopwords: question/function words that waste the token cap and dilute
// bm25 when OR'd. Content-bearing short words stay (ssh, db, gpu — and "up",
// "out", "get", which carry meaning in engineering prose: scale up, roll out).
var ftsStopwords = func() map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields(`the and for with that this these those from into onto
		over under about after before between what when where which who whom whose why how
		does did do doing done is are was were be been being has have had can could should
		would will shall may might must you your yours we our ours they them their it its
		he she his her in on at to of as by an or if so no not but there here then than
		too very just also any all some`) {
		m[w] = true
	}
	return m
}()

// ftsQuery ORs up to 12 content tokens. Stopwords are dropped first so long
// natural-language questions don't burn the cap before their rare terms; an
// all-stopword query falls back to the raw tokens rather than matching nothing.
func ftsQuery(text string) string {
	raw := tokenRe.FindAllString(text, 64)
	if len(raw) == 0 {
		return `""`
	}
	seen := map[string]bool{}
	var toks []string
	for _, t := range raw {
		lc := strings.ToLower(t)
		if ftsStopwords[lc] || seen[lc] {
			continue
		}
		seen[lc] = true
		toks = append(toks, t)
		if len(toks) == 12 {
			break
		}
	}
	if len(toks) == 0 {
		toks = raw
		if len(toks) > 12 {
			toks = toks[:12]
		}
	}
	quoted := make([]string, len(toks))
	for i, t := range toks {
		quoted[i] = `"` + t + `"`
	}
	return strings.Join(quoted, " OR ")
}

func ToFactOut(r FactRow, score, now float64, src string) FactOut {
	var ents, fls []string
	_ = json.Unmarshal([]byte(r.Entities), &ents)
	_ = json.Unmarshal([]byte(r.Files), &fls)
	return FactOut{
		ID: r.ID, Statement: r.Statement, Kind: r.Kind, Repo: r.Repo,
		Entities: ents, Files: fls, Confidence: r.Confidence,
		Score:   math.Round(score*1000) / 1000,
		Mass:    math.Round(massNow(r.Kind, r.Mass, r.ValidFrom, now)*1000) / 1000,
		AgeDays: math.Round((now-r.ValidFrom)/8640) / 10, Src: src,
		Stale:    isStale(r.Kind, r.ValidFrom, now),
		AsOfDate: store.AsOfDate(r.ValidFrom),
	}
}

// search fuses lexical (FTS/bm25) and semantic (cosine) rankings with RRF,
// then nudges by decayed mass, confidence, and repo match.
func Search(db *sql.DB, q, repo string, k int, asOf float64, reinforce bool) ([]FactOut, error) {
	now := float64(time.Now().Unix())
	rdb := store.ReadDB(db) // parallel-read pool; db stays the serialized writer

	// temporal intent (pure heuristics). An explicit as-of is already a
	// historical frame — never apply CURRENT freshness pressure there.
	intent := classifyTemporal(q)
	if asOf > 0 {
		intent = TemporalHistorical
	}

	// The four independent probe arms run concurrently and join in a fixed
	// order, so ranking is byte-identical to the sequential version:
	//   1. query embed (GPU/HTTP bound)
	//   2. live FTS (sqlite)
	//   3. dead FTS probe (sqlite; row fetch only — chain walk stays serial)
	//   4. paraphrase FTS (sqlite)
	// Long multi-term questions cost ~35ms per FTS arm, so overlap is the
	// difference between ~90ms and ~40ms. Errors stay loud at each join.
	type embedRes struct {
		qv  [][]float32
		err error
	}
	embedCh := make(chan embedRes, 1)
	go func() {
		qv, err := embedTexts([]string{q})
		embedCh <- embedRes{qv, err}
	}()

	type deadHit struct{ id, next int64 }
	type deadRes struct {
		dead []deadHit
		err  error
	}
	deadCh := make(chan deadRes, 1)
	if asOf == 0 {
		go func() {
			hrows, err := rdb.Query(`SELECT f.id, f.superseded_by FROM facts_fts
				JOIN facts f ON f.id = facts_fts.rowid
				WHERE facts_fts MATCH ? AND f.superseded_at IS NOT NULL
				ORDER BY bm25(facts_fts, 2.0, 1.0) LIMIT 10`, ftsQuery(q))
			if err != nil {
				deadCh <- deadRes{nil, err}
				return
			}
			var dead []deadHit
			for hrows.Next() {
				var d deadHit
				var nx sql.NullInt64
				if err := hrows.Scan(&d.id, &nx); err != nil {
					hrows.Close()
					deadCh <- deadRes{nil, err}
					return
				}
				d.next = nx.Int64
				dead = append(dead, d)
			}
			hrows.Close()
			deadCh <- deadRes{dead, nil}
		}()
	}

	type paraRes struct {
		rank map[int64]int
		err  error
	}
	paraCh := make(chan paraRes, 1)
	if asOf == 0 {
		go func() {
			rank, _, err := paraFtsRanks(rdb, q, 0)
			paraCh <- paraRes{rank, err}
		}()
	}

	// lexical arm
	// as-of runs on WORLD time: what was true at T, judged by when facts (and
	// their superseders) became true — not when oracle happened to ingest them.
	where := "WHERE facts_fts MATCH ?"
	args := []any{ftsQuery(q)}
	if asOf > 0 {
		where += " AND " + store.AsOfPredicate
		args = append(args, asOf, asOf)
	} else {
		where += " AND f.superseded_at IS NULL"
	}
	rows, err := rdb.Query(fmt.Sprintf(
		`SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid %s
		 ORDER BY bm25(facts_fts, 2.0, 1.0) LIMIT 60`, where), args...)
	if err != nil {
		return nil, err
	}
	ftsRank := map[int64]int{}
	var ftsOrder []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, err
		}
		ftsRank[id] = len(ftsOrder)
		ftsOrder = append(ftsOrder, id)
	}
	rows.Close()

	// chain continuity: old phrasings often match only the SUPERSEDED fact.
	// Follow those matches to their live chain heads and rank the head as if
	// it had matched (slightly discounted). Selfeval continuity measures this.
	viaHistory := map[int64]bool{}
	if asOf == 0 {
		dr := <-deadCh
		if dr.err != nil {
			return nil, dr.err
		}
		for rank, d := range dr.dead {
			head := d.next
			for hops := 0; head != 0 && hops < 20; hops++ { // walk to the live head
				var nx sql.NullInt64
				if err := rdb.QueryRow("SELECT superseded_by FROM facts WHERE id = ? AND superseded_at IS NOT NULL", head).Scan(&nx); err != nil {
					break // not superseded => live head
				}
				head = nx.Int64
			}
			if head == 0 {
				continue
			}
			if _, already := ftsRank[head]; !already {
				ftsRank[head] = len(ftsOrder) + rank + 2 // discounted vs direct matches
				ftsOrder = append(ftsOrder, head)
				viaHistory[head] = true
			}
		}
	}

	// paraphrase arm (third RRF source): the same fact under alternate wording.
	// Weighted 0.8 vs the primary lexical arm — a paraphrase match is real
	// signal but one step removed from the canonical statement.
	// Skipped in a historical frame: paraphrase coverage is live-only
	// (paraphraseRun selects superseded_at IS NULL), so under as-of the arm
	// would boost only the facts that happened to survive to today and crowd
	// era-correct superseded facts out of the top k.
	paraRank := map[int64]int{}
	if asOf == 0 {
		pr := <-paraCh
		if pr.err != nil {
			return nil, pr.err
		}
		paraRank = pr.rank
	}

	// semantic arm — degraded-off is not allowed (ADR-004 spirit): embed errors are loud
	er := <-embedCh
	if er.err != nil {
		return nil, fmt.Errorf("query embed: %w", er.err)
	}
	qv := er.qv

	// GPU vector store (daemon + ORACLE_MLX=1): three matmul+top-k calls
	// replace the SQL scans below. Present-time only — as-of keeps SQL.
	vs := activeVecStore()
	if vs != nil {
		if err := vs.refresh(rdb); err != nil {
			return nil, err
		}
	}

	var cosSim map[int64]float64
	var cosOrder []int64
	switch {
	case vs != nil && asOf == 0:
		cosSim, cosOrder, err = vs.cosineTop(qv[0], 60)
	case vs != nil:
		cosSim, cosOrder, err = vs.cosineTopAsOf(rdb, qv[0], asOf, 60)
	default:
		cosSim, cosOrder, err = cosineTop(rdb, qv[0], asOf, 60)
	}
	if err != nil {
		return nil, err
	}

	// paraphrase vectors: an alternate surface form of the SAME live fact. If
	// the paraphrase is semantically closer to the query than the canonical
	// statement, rank the fact by that similarity (no discount — same fact).
	// Skipped under as-of for the same live-only-coverage reason as para FTS.
	paraSim := map[int64]float64{}
	if asOf == 0 {
		if vs != nil {
			paraSim, err = vs.paraCosTop(qv[0], 60)
		} else {
			paraSim, err = paraCosTop(rdb, qv[0], 0, 60)
		}
		if err != nil {
			return nil, err
		}
	}
	{
		merged := false
		for id, sim := range paraSim {
			if sim > cosSim[id] {
				cosSim[id] = sim
				merged = true
			}
		}
		if merged {
			cosOrder = cosOrder[:0]
			for id := range cosSim {
				cosOrder = append(cosOrder, id)
			}
			sort.Slice(cosOrder, func(i, j int) bool { return cosSim[cosOrder[i]] > cosSim[cosOrder[j]] })
			if len(cosOrder) > 60 {
				for _, id := range cosOrder[60:] {
					delete(cosSim, id)
				}
				cosOrder = cosOrder[:60]
			}
		}
	}

	// chain continuity, cosine arm: an old phrasing is often a near-perfect
	// semantic match for the SUPERSEDED fact only — the live head is worded too
	// differently for either direct arm. Probe superseded facts, walk each to
	// its live head, and rank the head by the dead match's similarity
	// (discounted) so it lands near the top of the cosine ranking instead of
	// bolted to the tail like the FTS-side follow.
	deadSimByHead := map[int64]float64{}
	if asOf == 0 {
		var deadCos []deadCosHit
		if vs != nil {
			deadCos, err = vs.cosineDeadTop(qv[0], 10)
		} else {
			deadCos, err = cosineDeadTop(rdb, qv[0], 10)
		}
		if err != nil {
			return nil, err
		}
		inserted := false
		for _, d := range deadCos {
			head := d.Next
			for hops := 0; head != 0 && hops < 20; hops++ { // walk to the live head
				var nx sql.NullInt64
				if err := rdb.QueryRow("SELECT superseded_by FROM facts WHERE id = ? AND superseded_at IS NOT NULL", head).Scan(&nx); err != nil {
					break // not superseded => live head
				}
				head = nx.Int64
			}
			if head == 0 {
				continue
			}
			if d.Sim > deadSimByHead[head] {
				deadSimByHead[head] = d.Sim
			}
			if sim := d.Sim * 0.9; sim > cosSim[head] { // discount vs a direct match
				cosSim[head] = sim
				viaHistory[head] = true
				inserted = true
			}
		}
		if inserted { // re-rank by similarity with the followed heads in place
			cosOrder = cosOrder[:0]
			for id := range cosSim {
				cosOrder = append(cosOrder, id)
			}
			sort.Slice(cosOrder, func(i, j int) bool { return cosSim[cosOrder[i]] > cosSim[cosOrder[j]] })
			if len(cosOrder) > 60 {
				for _, id := range cosOrder[60:] {
					delete(cosSim, id)
				}
				cosOrder = cosOrder[:60]
			}
		}
	}
	cosRank := map[int64]int{}
	for i, id := range cosOrder {
		cosRank[id] = i
	}

	// union + RRF
	const rrfK = 60.0
	ids := map[int64]bool{}
	for _, id := range ftsOrder {
		ids[id] = true
	}
	for _, id := range cosOrder {
		ids[id] = true
	}
	for id := range paraRank {
		ids[id] = true
	}
	if len(ids) == 0 {
		return []FactOut{}, nil
	}
	ph := strings.TrimSuffix(strings.Repeat("?,", len(ids)), ",")
	fargs := make([]any, 0, len(ids))
	for id := range ids {
		fargs = append(fargs, id)
	}
	frows, err := rdb.Query(`SELECT id, statement, kind, COALESCE(repo,''), entities, files,
		confidence, mass, recorded_at, valid_from, COALESCE(src_session,''),
		corroborations, evidence,
		(SELECT COUNT(*) FROM edges ce JOIN facts cf ON cf.id = ce.src
		 WHERE ce.dst = facts.id AND ce.type = 'contradicts' AND cf.superseded_at IS NULL) AS ncontra
		FROM facts WHERE id IN (`+ph+`)`, fargs...)
	if err != nil {
		return nil, err
	}
	defer frows.Close()
	type scored struct {
		s       float64
		r       FactRow
		src     string
		corrob  int
		evid    string
		ncontra int
	}
	var all []scored
	ref := now
	if asOf > 0 {
		ref = asOf
	}
	for frows.Next() {
		var r FactRow
		var src, evid string
		var corrob, ncontra int
		if err := frows.Scan(&r.ID, &r.Statement, &r.Kind, &r.Repo, &r.Entities, &r.Files,
			&r.Confidence, &r.Mass, &r.RecordedAt, &r.ValidFrom, &src, &corrob, &evid, &ncontra); err != nil {
			return nil, err
		}
		s := 0.0
		if rk, ok := ftsRank[r.ID]; ok {
			s += 1.0 / (rrfK + float64(rk))
		}
		if rk, ok := cosRank[r.ID]; ok {
			s += 1.2 / (rrfK + float64(rk)) // semantic arm weighted slightly up
			s += 0.004 * cosSim[r.ID]       // break ties toward true similarity
		}
		if rk, ok := paraRank[r.ID]; ok {
			s += 0.8 / (rrfK + float64(rk)) // paraphrase arm, discounted vs canonical lexical
		}
		// temporal typing: a CURRENT-intent query cares about freshness of
		// perishable kinds — triple the decay-mass term, and hard-demote (never
		// hide) perishables past 3x their half-life below any fresher hit.
		massW := 0.004
		if intent == TemporalCurrent && perishableKinds[r.Kind] {
			massW = 0.012
			if isRotten(r.Kind, r.ValidFrom, ref) {
				s -= 1.0 // score penalty, not a filter: still returned, labeled stale
			}
		}
		s += massW*math.Log(massNow(r.Kind, r.Mass, r.ValidFrom, ref)) + 0.003*r.Confidence
		if repo != "" && r.Repo == repo {
			s += 0.008
		}
		// truth prior: independent corroboration lifts, live contradictions and
		// uncorroborated bare assertions sink (anti over-index on one agent's claim)
		s += 0.003 * math.Log1p(float64(corrob))
		s -= 0.006 * float64(ncontra)
		if evid == "asserted" && corrob == 0 {
			s -= 0.002
		}
		// continuity: a live head whose superseded ancestor is a strong semantic
		// match for the query IS the answer to that (old) phrasing — single-arm
		// RRF credit alone loses to two-arm near-duplicates in dense topics
		if ds := deadSimByHead[r.ID]; ds > 0 {
			s += 0.008 * ds
		}
		all = append(all, scored{s, r, src, corrob, evid, ncontra})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].s > all[j].s })
	if len(all) > k {
		all = all[:k]
	}
	out := make([]FactOut, 0, len(all))
	var hitIDs []int64
	for _, sc := range all {
		fo := ToFactOut(sc.r, sc.s, now, sc.src)
		fo.Evidence, fo.Corrob, fo.ViaHistory = sc.evid, sc.corrob, viaHistory[sc.r.ID]
		if sc.ncontra > 0 {
			crows, _ := rdb.Query(`SELECT ce.src FROM edges ce JOIN facts cf ON cf.id = ce.src
				WHERE ce.dst = ? AND ce.type = 'contradicts' AND cf.superseded_at IS NULL LIMIT 5`, sc.r.ID)
			if crows != nil {
				for crows.Next() {
					var cid int64
					_ = crows.Scan(&cid)
					fo.ContraBy = append(fo.ContraBy, cid)
				}
				crows.Close()
			}
		}
		out = append(out, fo)
		hitIDs = append(hitIDs, sc.r.ID)
	}
	if reinforce && asOf == 0 && len(hitIDs) > 0 {
		// best-effort write (errors already ignored inside); run it off the
		// response path so the WAL fsync never adds to query latency.
		top := append([]int64(nil), hitIDs[:min(3, len(hitIDs))]...)
		go ReinforceFacts(db, top)
	}
	// trace the ranking evidence, not just the ids: per-hit
	// [ftsRank, cosRank, cosSim, decayedMass, confidence, repoMatch] is the
	// replay buffer for fitting the hand-tuned scoring weights offline later.
	feats := map[string][]float64{}
	for _, sc := range all {
		fr, cr := -1.0, -1.0
		if r, ok := ftsRank[sc.r.ID]; ok {
			fr = float64(r)
		}
		if r, ok := cosRank[sc.r.ID]; ok {
			cr = float64(r)
		}
		rm := 0.0
		if repo != "" && sc.r.Repo == repo {
			rm = 1
		}
		feats[strconv.FormatInt(sc.r.ID, 10)] = []float64{fr, cr,
			math.Round(cosSim[sc.r.ID]*1000) / 1000,
			math.Round(massNow(sc.r.Kind, sc.r.Mass, sc.r.ValidFrom, ref)*1000) / 1000,
			sc.r.Confidence, rm}
	}
	resJSON, _ := json.Marshal(map[string]any{"ids": hitIDs, "feat": feats})
	_, _ = db.Exec("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)", now, "query", q, string(resJSON))
	return out, nil
}

// reinforceFacts bumps retrieval mass (capped) and usage counters. Called for
// facts a human query surfaced, or that an ask answer actually cited — not for
// everything a reasoner glanced at, which would compound rich-get-richer.
// Telemetry, like traces: best-effort by design — a failed counter bump must
// never fail a search or eat an answer.
func ReinforceFacts(db *sql.DB, ids []int64) {
	if len(ids) == 0 {
		return
	}
	ph := strings.TrimSuffix(strings.Repeat("?,", len(ids)), ",")
	args := make([]any, 0, len(ids)+1)
	args = append(args, float64(time.Now().Unix()))
	for _, id := range ids {
		args = append(args, id)
	}
	_, _ = db.Exec("UPDATE facts SET mass = min(mass + 0.15, 3.0), use_count = use_count + 1, last_used_at = ? WHERE id IN ("+ph+")", args...)
}

func Brief(db *sql.DB, repo string, k int) (map[string][]FactOut, error) {
	now := float64(time.Now().Unix())
	where, args := "WHERE superseded_at IS NULL", []any{}
	if repo != "" {
		where += " AND repo = ?"
		args = append(args, repo)
	}
	rows, err := db.Query(`SELECT id, statement, kind, COALESCE(repo,''), entities, files,
		confidence, mass, recorded_at, valid_from, COALESCE(src_session,'') FROM facts `+where, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type scored struct {
		s       float64
		r       FactRow
		src     string
		corrob  int
		evid    string
		ncontra int
	}
	var all []scored
	for rows.Next() {
		var r FactRow
		var src string
		if err := rows.Scan(&r.ID, &r.Statement, &r.Kind, &r.Repo, &r.Entities, &r.Files,
			&r.Confidence, &r.Mass, &r.RecordedAt, &r.ValidFrom, &src); err != nil {
			return nil, err
		}
		all = append(all, scored{s: massNow(r.Kind, r.Mass, r.ValidFrom, now) * (0.5 + r.Confidence), r: r, src: src})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].s > all[j].s })
	if len(all) > k {
		all = all[:k]
	}
	out := map[string][]FactOut{}
	for _, sc := range all {
		out[sc.r.Kind] = append(out[sc.r.Kind], ToFactOut(sc.r, sc.s, now, sc.src))
	}
	return out, nil
}

func EntityView(db *sql.DB, name string, k int) (map[string]any, error) {
	// canonical resolver: exact name, then live alias (post-merge names), then prefix
	id, display, ok := kb.ResolveEntity(db, name)
	if !ok {
		return map[string]any{"entity": nil}, nil
	}
	var seen int
	_ = db.QueryRow("SELECT seen_count FROM entities WHERE id = ?", id).Scan(&seen)
	now := float64(time.Now().Unix())
	rows, err := db.Query(`SELECT f.id, f.statement, f.kind, COALESCE(f.repo,''), f.entities, f.files,
		f.confidence, f.mass, f.recorded_at, f.valid_from, COALESCE(f.src_session,'')
		FROM fact_entities fe JOIN facts f ON f.id = fe.fact_id
		WHERE fe.entity_id = ? AND f.superseded_at IS NULL`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var facts []FactOut
	for rows.Next() {
		var r FactRow
		var src string
		if err := rows.Scan(&r.ID, &r.Statement, &r.Kind, &r.Repo, &r.Entities, &r.Files,
			&r.Confidence, &r.Mass, &r.RecordedAt, &r.ValidFrom, &src); err != nil {
			return nil, err
		}
		facts = append(facts, ToFactOut(r, massNow(r.Kind, r.Mass, r.ValidFrom, now), now, src))
	}
	sort.Slice(facts, func(i, j int) bool { return facts[i].Mass > facts[j].Mass })
	if len(facts) > k {
		facts = facts[:k]
	}
	co, err := db.Query(`SELECT en.display, ee.count FROM entity_edges ee
		JOIN entities en ON en.id = CASE WHEN ee.a = ? THEN ee.b ELSE ee.a END
		WHERE ee.a = ? OR ee.b = ? ORDER BY ee.count DESC LIMIT 15`, id, id, id)
	if err != nil {
		return nil, err
	}
	defer co.Close()
	var comention []map[string]any
	for co.Next() {
		var n string
		var c int
		_ = co.Scan(&n, &c)
		comention = append(comention, map[string]any{"name": n, "count": c})
	}
	return map[string]any{"entity": display, "seen_count": seen, "facts": facts, "co_mentioned": comention}, nil
}

func linkEntities(tx *sql.Tx, factID int64, names []string, now float64) error {
	var ids []int64
	for _, n := range names {
		if !kb.ValidEntityName(strings.ToLower(strings.TrimSpace(n))) {
			continue
		}
		// alias- and variant-aware: a mention of a merged-away or respelled
		// name links to the canonical entity instead of minting a duplicate
		eid, err := kb.ResolveOrCreateEntity(tx, n, "", now)
		if err != nil {
			return err
		}
		if _, err := tx.Exec("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES(?,?)", factID, eid); err != nil {
			return err
		}
		ids = append(ids, eid)
	}
	for i, a := range ids {
		for _, b := range ids[i+1:] {
			lo, hi := min(a, b), max(a, b)
			if _, err := tx.Exec(`INSERT INTO entity_edges(a, b, count, last_seen) VALUES(?,?,1,?)
				ON CONFLICT(a, b) DO UPDATE SET count = count + 1, last_seen = ?`, lo, hi, now, now); err != nil {
				return err
			}
		}
	}
	return nil
}

func supersedeCandidates(db *sql.DB, f ingest.Fact, repo string) ([]ingest.OldFact, error) {
	// scope: same repo, OR any fact sharing an entity with the new one — a fact
	// learned in another repo (or under repo "unknown") must still be able to
	// outdate a stale fact about the same system. Exact/alias resolution only:
	// the prefix fallback is a human-input convenience and has no place here.
	var eids []any
	for _, n := range f.Entities {
		if id, _, ok := kb.ResolveEntityExact(db, n); ok {
			eids = append(eids, id)
		}
	}
	scope := "f.repo = ?"
	args := []any{ftsQuery(f.Statement), f.Kind, repo}
	if len(eids) > 0 {
		ph := strings.TrimSuffix(strings.Repeat("?,", len(eids)), ",")
		scope = "(f.repo = ? OR f.id IN (SELECT fact_id FROM fact_entities WHERE entity_id IN (" + ph + ")))"
		args = append(args, eids...)
	}
	rows, err := db.Query(`SELECT f.id, f.statement, f.valid_from, f.evidence FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid
		WHERE facts_fts MATCH ? AND f.superseded_at IS NULL AND f.kind = ? AND `+scope+`
		ORDER BY bm25(facts_fts) LIMIT 3`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ingest.OldFact
	for rows.Next() {
		var o ingest.OldFact
		if err := rows.Scan(&o.ID, &o.Statement, &o.ValidFrom, &o.Evidence); err != nil {
			return nil, err
		}
		out = append(out, o)
	}
	return out, nil
}

// ingestChunk extracts + upserts one chunk. Offset advance commits atomically with facts.
func ingestChunk(db *sql.DB, c ingest.Chunk) (int, error) {
	var facts []ingest.Fact
	var verdicts, conflicts map[int][]int64
	if strings.TrimSpace(c.Text) != "" {
		date := store.LocalDate(c.EventTime)
		var err error
		facts, err = ingest.ExtractFacts(c.Text, c.Repo, date)
		if err != nil {
			return 0, err
		}
		cands := map[int][]ingest.OldFact{}
		for i, f := range facts {
			cc, err := supersedeCandidates(db, f, c.Repo)
			if err != nil {
				return 0, err
			}
			if len(cc) > 0 {
				cands[i] = cc
			}
		}
		verdicts, conflicts, err = ingest.JudgeSupersede(db, facts, cands, date)
		if err != nil {
			return 0, err
		}
	}
	now := float64(time.Now().Unix())
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	for i, f := range facts {
		// re-statement (exact OR formatting-variant: parallel subagents restate
		// the same finding with cosmetic drift, matched by stmt_hash):
		// corroborate rather than duplicate — but ONLY from a different session
		// (an agent repeating itself earns nothing), and it refreshes the decay
		// clock (re-asserted => still true).
		h := store.StmtHash(f.Statement)
		var dup int64
		var dupSess string
		if tx.QueryRow(`SELECT id, COALESCE(src_session,'') FROM facts
			WHERE stmt_hash = ? AND repo = ? AND superseded_at IS NULL`,
			h, c.Repo).Scan(&dup, &dupSess) == nil {
			if dupSess != c.Session {
				if _, err := tx.Exec(`UPDATE facts SET corroborations = corroborations + 1,
					confidence = min(0.98, 1.0 - (1.0 - confidence)*0.7),
					valid_from = max(valid_from, ?),
					evidence = CASE WHEN ? = 'verified' THEN 'verified' ELSE evidence END
					WHERE id = ?`, c.EventTime, f.Evidence, dup); err != nil {
					return 0, err
				}
			}
			continue
		}
		ents, _ := json.Marshal(f.Entities)
		fls, _ := json.Marshal(f.Files)
		res, err := tx.Exec(`INSERT INTO facts(statement, kind, repo, entities, files, confidence,
			recorded_at, valid_from, src_path, src_session, evidence, stmt_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`,
			f.Statement, f.Kind, c.Repo, string(ents), string(fls), f.Confidence,
			now, c.EventTime, c.Path, c.Session, f.Evidence, h)
		if err != nil {
			return 0, err
		}
		newID, _ := res.LastInsertId()
		if err := linkEntities(tx, newID, f.Entities, now); err != nil {
			return 0, err
		}
		for _, oldID := range conflicts[i] {
			if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
				VALUES(?,?,?,?)`, newID, oldID, "contradicts", now); err != nil {
				return 0, err
			}
		}
		for _, oldID := range verdicts[i] {
			if _, err := tx.Exec(`UPDATE facts SET superseded_at = ?, superseded_by = ?
				WHERE id = ? AND superseded_at IS NULL`, now, newID, oldID); err != nil {
				return 0, err
			}
			if _, err := tx.Exec(`INSERT OR IGNORE INTO edges(src, dst, type, recorded_at)
				VALUES(?,?,?,?)`, newID, oldID, "supersedes", now); err != nil {
				return 0, err
			}
		}
	}
	if strings.TrimSpace(c.Text) != "" {
		// screener/rewrite training substrate: every judged chunk + its yield
		if _, err := tx.Exec(`INSERT INTO chunk_log(ts, path, source, repo, chars, n_facts, text)
			VALUES(?,?,?,?,?,?,?)`, now, c.Path, c.Source, c.Repo, len(c.Text), len(facts), c.Text); err != nil {
			return 0, err
		}
	}
	if _, err := tx.Exec(`INSERT INTO files(path, source, repo, offset, mtime, last_scan) VALUES(?,?,?,?,?,?)
		ON CONFLICT(path) DO UPDATE SET offset = excluded.offset, repo = excluded.repo,
		mtime = excluded.mtime, last_scan = excluded.last_scan, error_count = 0, last_error_ts = 0`,
		c.Path, c.Source, c.Repo, c.EndOffset, c.EventTime, now); err != nil {
		return 0, err
	}
	return len(facts), tx.Commit()
}

type CycleStats struct {
	FilesSeen     int    `json:"files_seen"`
	Chunks        int    `json:"chunks"`
	Facts         int    `json:"facts"`
	Errors        int    `json:"errors"`
	SkippedBudget bool   `json:"skipped_budget"`
	Embedded      int    `json:"embedded"`
	Enriched      int    `json:"enriched"`
	Backoff       int    `json:"backoff"`        // files currently inside their error-backoff window
	EnrichPending int    `json:"enrich_pending"` // facts not yet in enrich_done
	TracesPruned  int64  `json:"traces_pruned,omitempty"`
	Backup        string `json:"backup,omitempty"`
	LastError     string `json:"last_error,omitempty"`
}

// cycle runs one ingest pass. maxCalls caps extraction LLM calls (cost knob).
// forceRetry ignores per-file error backoff for this pass (drain the tail);
// files in their backoff window are still counted in stats.Backoff either way.
func Cycle(db *sql.DB, maxCalls int, sinceDays float64, workers int, forceRetry bool) (CycleStats, error) {
	if workers < 1 {
		workers = 4
	}
	var st CycleStats
	calls := 0
	nowTs := float64(time.Now().Unix())

	// lease: one cycle at a time across processes (daemon + manual runs)
	res, err := db.Exec(`INSERT INTO meta(k, v) VALUES('cycle_lease', ?)
		ON CONFLICT(k) DO UPDATE SET v = excluded.v WHERE CAST(meta.v AS REAL) < ?`,
		fmt.Sprintf("%f", nowTs+1800), nowTs)
	if err != nil {
		return st, err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		st.LastError = "another cycle holds the lease; skipped"
		return st, nil
	}
	defer db.Exec("DELETE FROM meta WHERE k='cycle_lease'")

	type fileState struct {
		offset    int64
		repo      string
		errCount  int
		lastErrTs float64
	}
	known := map[string]fileState{}
	rows, err := db.Query("SELECT path, offset, COALESCE(repo,''), error_count, last_error_ts FROM files")
	if err != nil {
		return st, err
	}
	for rows.Next() {
		var fs fileState
		var p string
		_ = rows.Scan(&p, &fs.offset, &fs.repo, &fs.errCount, &fs.lastErrTs)
		known[p] = fs
	}
	rows.Close()

	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers) // parallel files; LLM calls dominate wall-clock
	for _, sf := range ingest.Discover(sinceDays) {
		k := known[sf.Path]
		fi, err := os.Stat(sf.Path)
		if err != nil || fi.Size() <= k.offset {
			continue
		}
		// exponential backoff for chunks that keep failing: 30min * 2^errors, cap 24h
		if k.errCount > 0 {
			backoff := 1800.0 * float64(int64(1)<<min(k.errCount, 6))
			if backoff > 86400 {
				backoff = 86400
			}
			if nowTs-k.lastErrTs < backoff {
				st.Backoff++ // debt gauge: still counted when force-retrying
				if !forceRetry {
					continue
				}
			}
		}
		st.FilesSeen++
		repo := ""
		if k.repo != "" && k.repo != "unknown" {
			repo = k.repo
		}
		wg.Add(1)
		go func(sf ingest.SessionFile, offset int64, repo string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			chunks, err := ingest.ReadNew(sf.Path, sf.Source, offset, repo)
			if err != nil {
				mu.Lock()
				st.Errors++
				st.LastError = err.Error()
				mu.Unlock()
				return
			}
			// chunks within a file stay sequential: later facts may supersede earlier ones
			for _, c := range chunks {
				if strings.TrimSpace(c.Text) != "" {
					mu.Lock()
					if calls >= maxCalls {
						st.SkippedBudget = true
						mu.Unlock()
						return
					}
					calls++
					mu.Unlock()
				}
				n, err := ingestChunk(db, c)
				mu.Lock()
				if err != nil {
					st.Errors++
					st.LastError = err.Error()
					mu.Unlock()
					_, _ = db.Exec("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
						float64(time.Now().Unix()), "error", c.Path, store.Truncate(err.Error(), 500))
					_, _ = db.Exec(`INSERT INTO files(path, source, repo, offset, error_count, last_error_ts)
						VALUES(?,?,?,?,1,?) ON CONFLICT(path) DO UPDATE SET
						error_count = error_count + 1, last_error_ts = excluded.last_error_ts`,
						c.Path, c.Source, c.Repo, offset, float64(time.Now().Unix()))
					return // do not advance past a failed chunk in this file
				}
				st.Facts += n
				st.Chunks++
				mu.Unlock()
			}
		}(sf, k.offset, repo)
	}
	wg.Wait()
	if n, err := ingest.EnrichSome(db, 3); err != nil { // steady-state trickle; bulk via `oracle enrich`
		st.Errors++
		st.LastError = "enrich: " + err.Error()
	} else {
		st.Enriched = n
	}
	emb, err := embedMissing(db, 32)
	st.Embedded = emb
	if err != nil {
		st.Errors++
		st.LastError = "embed: " + err.Error()
	}
	if n, err := store.PruneTraces(db, store.TracesKeep); err != nil {
		st.Errors++
		st.LastError = "prune traces: " + err.Error()
	} else {
		st.TracesPruned = n
	}
	// remaining-debt gauge: enrichment backlog (files in backoff are st.Backoff)
	if err := db.QueryRow(`SELECT COUNT(*) FROM facts f LEFT JOIN enrich_done d ON d.fact_id = f.id
		WHERE d.fact_id IS NULL`).Scan(&st.EnrichPending); err != nil {
		st.Errors++
		st.LastError = "enrich_pending gauge: " + err.Error()
	}
	if dest, err := store.BackupIfDue(db); err != nil {
		st.Errors++
		st.LastError = "backup: " + err.Error()
	} else if dest != "" {
		st.Backup = dest
	}
	// nightly selfeval: catch retrieval regressions the day they land
	var lastEval float64
	_ = db.QueryRow("SELECT CAST(v AS REAL) FROM meta WHERE k='last_selfeval_ts'").Scan(&lastEval)
	if float64(time.Now().Unix())-lastEval > 22*3600 {
		if ev, err := RunSelfEval(db, 25, 10, false); err == nil {
			eb, _ := json.Marshal(ev)
			_, _ = db.Exec("INSERT OR REPLACE INTO meta(k, v) VALUES('last_selfeval', ?)", string(eb))
			_, _ = db.Exec("INSERT OR REPLACE INTO meta(k, v) VALUES('last_selfeval_ts', ?)",
				fmt.Sprintf("%d", time.Now().Unix()))
			_, _ = db.Exec("INSERT INTO traces(ts, kind, q, results) VALUES(?,?,?,?)",
				float64(time.Now().Unix()), "selfeval", "", string(eb))
		}
	}
	b, _ := json.Marshal(map[string]any{"ts": time.Now().Unix(), "stats": st})
	_, _ = db.Exec("INSERT OR REPLACE INTO meta(k, v) VALUES('last_cycle', ?)", string(b))
	return st, nil
}
