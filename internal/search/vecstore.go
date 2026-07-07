package search

// vecstore.go — memory-resident cosine search for the three semantic arms.
//
// The daemon (`oracle up`) loads all corpus vectors into a vector engine once
// at start: three named sets — live facts, superseded facts, live-fact
// paraphrases — plus a Go-side fact_id map per set. search() then replaces
// the three per-query SQL scans + blob decodes + Go cosine loops with one
// matmul+top-k per arm. The engine is Metal fp16 under ORACLE_MLX=1
// (infer_mlx_darwin.go) and plain parallel Go everywhere else (veccpu.go) —
// same contract, same ranking.
//
// Freshness: a cheap signature (row counts + max ids + supersede count) is
// checked before every search; any change (ingest cycle, reembed, supersede)
// reloads the affected world in one shot. CLI one-shot queries never build the
// store (initVecStore is only called from serve) and keep the SQL path. As-of
// queries engine-score the full corpus and apply the validity-at-T filter
// host-side (cosineTopAsOf), so historical frames stay exact.
//
// ORACLE_VECS=0 disables the store; ORACLE_MLX_VECS=0 is honored as the
// legacy spelling of the same kill switch.

import (
	"database/sql"
	"fmt"
	"log"
	"oracle/internal/store"
	"os"
	"sort"
	"sync"
	"time"
)

const (
	vecSetLive = "facts_live"
	vecSetDead = "facts_dead"
	vecSetPara = "facts_para"
	vecSetAll  = "facts_all" // live + dead, for as-of (host-side validity filter)
)

type vecStore struct {
	mu       sync.Mutex
	sig      string
	sigAt    time.Time // last signature check; re-checked at most every sigTTL
	liveIDs  []int64
	deadIDs  []int64
	deadNext []int64 // superseded_by per deadIDs row (0 if orphaned)
	paraIDs  []int64
	allIDs   []int64
}

// sigTTL bounds how stale the GPU sets may be vs the DB. The signature query
// itself costs ~35ms (COUNTs over blob-heavy tables), so it must not run per
// query; ingest lands in 300s cycles, so a few seconds of lag is invisible.
const sigTTL = 3 * time.Second

var gVecStore *vecStore

// activeVecStore returns the daemon's GPU store, or nil (CLI / disabled).
func activeVecStore() *vecStore { return gVecStore }

// initVecStore builds the store; call once from the daemon after the DB is
// open. Loud failure: a broken store must not silently degrade to the slow
// path.
func InitVecStore(db *sql.DB) error {
	if os.Getenv("ORACLE_VECS") == "0" || os.Getenv("ORACLE_MLX_VECS") == "0" {
		return nil
	}
	if err := checkLocalVecs(db); err != nil {
		return err
	}
	s := &vecStore{}
	if err := s.reload(db); err != nil {
		return fmt.Errorf("vecstore: %w", err)
	}
	gVecStore = s
	return nil
}

// signature captures everything that can change set membership or content:
// vec row counts, max fact ids (append), and the supersede count (live<->dead
// moves). One indexed-subquery row fetch, run before every search.
func vecSig(db *sql.DB) (string, error) {
	q := `SELECT
		(SELECT COUNT(*) FROM ` + factVecsTable() + `),
		(SELECT COALESCE(MAX(fact_id),0) FROM ` + factVecsTable() + `),
		(SELECT COUNT(*) FROM facts WHERE superseded_at IS NOT NULL),
		(SELECT COUNT(*) FROM ` + paraVecsTable() + `),
		(SELECT COALESCE(MAX(fact_id),0) FROM ` + paraVecsTable() + `)`
	var a, b, c, d, e int64
	if err := db.QueryRow(q).Scan(&a, &b, &c, &d, &e); err != nil {
		return "", fmt.Errorf("vecstore: signature: %w", err)
	}
	return fmt.Sprintf("%d/%d/%d/%d/%d", a, b, c, d, e), nil
}

// refresh reloads the GPU sets iff the corpus changed since the last load;
// the signature is re-checked at most every sigTTL.
func (s *vecStore) refresh(db *sql.DB) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if time.Since(s.sigAt) < sigTTL {
		return nil
	}
	sig, err := vecSig(db)
	if err != nil {
		return err
	}
	s.sigAt = time.Now()
	if sig == s.sig {
		return nil
	}
	return s.reloadLocked(db, sig)
}

func (s *vecStore) reload(db *sql.DB) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sig, err := vecSig(db)
	if err != nil {
		return err
	}
	s.sigAt = time.Now()
	return s.reloadLocked(db, sig)
}

func (s *vecStore) reloadLocked(db *sql.DB, sig string) error {
	start := time.Now()
	liveIDs, _, liveFlat, err := loadVecRows(db,
		"SELECT v.fact_id, 0, v.vec FROM "+factVecsTable()+
			" v JOIN facts f ON f.id = v.fact_id WHERE f.superseded_at IS NULL ORDER BY v.fact_id")
	if err != nil {
		return err
	}
	deadIDs, deadNext, deadFlat, err := loadVecRows(db,
		"SELECT v.fact_id, COALESCE(f.superseded_by, 0), v.vec FROM "+factVecsTable()+
			" v JOIN facts f ON f.id = v.fact_id WHERE f.superseded_at IS NOT NULL ORDER BY v.fact_id")
	if err != nil {
		return err
	}
	paraIDs, _, paraFlat, err := loadVecRows(db,
		"SELECT v.fact_id, 0, v.vec FROM "+paraVecsTable()+
			" v JOIN facts f ON f.id = v.fact_id WHERE f.superseded_at IS NULL ORDER BY v.fact_id")
	if err != nil {
		return err
	}
	allIDs, _, allFlat, err := loadVecRows(db,
		"SELECT v.fact_id, 0, v.vec FROM "+factVecsTable()+
			" v JOIN facts f ON f.id = v.fact_id ORDER BY v.fact_id")
	if err != nil {
		return err
	}
	if err := vecsLoad(vecSetLive, liveFlat, len(liveIDs), embedDims); err != nil {
		return err
	}
	if err := vecsLoad(vecSetDead, deadFlat, len(deadIDs), embedDims); err != nil {
		return err
	}
	if err := vecsLoad(vecSetPara, paraFlat, len(paraIDs), embedDims); err != nil {
		return err
	}
	if err := vecsLoad(vecSetAll, allFlat, len(allIDs), embedDims); err != nil {
		return err
	}
	s.liveIDs, s.deadIDs, s.deadNext, s.paraIDs, s.allIDs = liveIDs, deadIDs, deadNext, paraIDs, allIDs
	s.sig = sig
	log.Printf("vecstore: loaded %d live + %d dead + %d para vectors to %s engine in %dms",
		len(liveIDs), len(deadIDs), len(paraIDs), vecEngineLabel(), time.Since(start).Milliseconds())
	return nil
}

// loadVecRows reads (fact_id, aux, vec-blob) rows into parallel id/aux slices
// and one flat float32 buffer. Vector width must be embedDims — anything else
// is corrupt and fails loudly.
func loadVecRows(db *sql.DB, q string) ([]int64, []int64, []float32, error) {
	rows, err := db.Query(q)
	if err != nil {
		return nil, nil, nil, err
	}
	defer rows.Close()
	var ids, aux []int64
	var flat []float32
	for rows.Next() {
		var id, a int64
		var blob []byte
		if err := rows.Scan(&id, &a, &blob); err != nil {
			return nil, nil, nil, err
		}
		if len(blob) != 4*embedDims {
			return nil, nil, nil, fmt.Errorf("vecstore: fact %d vec is %d bytes, want %d", id, len(blob), 4*embedDims)
		}
		ids = append(ids, id)
		aux = append(aux, a)
		flat = append(flat, store.BlobToVec(blob)...)
	}
	return ids, aux, flat, rows.Err()
}

// topk maps a set's GPU result rows back to fact ids.
func (s *vecStore) topk(set string, ids []int64, qv []float32, n int) ([]int64, []float32, error) {
	if len(ids) == 0 {
		return nil, nil, nil
	}
	if n > len(ids) {
		n = len(ids)
	}
	idx, scores, err := vecsTopK(set, qv, n)
	if err != nil {
		return nil, nil, err
	}
	out := make([]int64, len(idx))
	for i, j := range idx {
		if int(j) >= len(ids) {
			return nil, nil, fmt.Errorf("vecstore: %s returned row %d beyond %d ids", set, j, len(ids))
		}
		out[i] = ids[j]
	}
	return out, scores, nil
}

// cosineTop: GPU twin of embed.go cosineTop (present-time only).
func (s *vecStore) cosineTop(qv []float32, n int) (map[int64]float64, []int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids, scores, err := s.topk(vecSetLive, s.liveIDs, qv, n)
	if err != nil {
		return nil, nil, err
	}
	m := make(map[int64]float64, len(ids))
	for i, id := range ids {
		m[id] = float64(scores[i])
	}
	return m, ids, nil
}

// cosineTopAsOf: GPU twin of embed.go cosineTop with asOf > 0. The GPU does
// every dot product (the expensive half); validity-at-T comes from one SQL id
// scan and the top-n select happens host-side — same ranking as the SQL path.
func (s *vecStore) cosineTopAsOf(db *sql.DB, qv []float32, asOf float64, n int) (map[int64]float64, []int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.allIDs) == 0 {
		return map[int64]float64{}, nil, nil
	}
	scores, err := vecsScore(vecSetAll, qv, len(s.allIDs))
	if err != nil {
		return nil, nil, err
	}
	rows, err := db.Query("SELECT f.id FROM facts f WHERE "+store.AsOfPredicate, asOf, asOf)
	if err != nil {
		return nil, nil, err
	}
	valid := map[int64]bool{}
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, nil, err
		}
		valid[id] = true
	}
	rows.Close()
	type pair struct {
		id  int64
		sim float64
	}
	var all []pair
	for i, id := range s.allIDs {
		if valid[id] {
			all = append(all, pair{id, float64(scores[i])})
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].sim > all[j].sim })
	if len(all) > n {
		all = all[:n]
	}
	m := make(map[int64]float64, len(all))
	order := make([]int64, 0, len(all))
	for _, p := range all {
		m[p.id] = p.sim
		order = append(order, p.id)
	}
	return m, order, nil
}

// cosineDeadTop: GPU twin of embed.go cosineDeadTop.
func (s *vecStore) cosineDeadTop(qv []float32, n int) ([]deadCosHit, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.deadIDs) == 0 {
		return nil, nil
	}
	if n > len(s.deadIDs) {
		n = len(s.deadIDs)
	}
	idx, scores, err := vecsTopK(vecSetDead, qv, n)
	if err != nil {
		return nil, err
	}
	hits := make([]deadCosHit, len(idx))
	for i, j := range idx {
		if int(j) >= len(s.deadIDs) {
			return nil, fmt.Errorf("vecstore: %s returned row %d beyond %d ids", vecSetDead, j, len(s.deadIDs))
		}
		hits[i] = deadCosHit{ID: s.deadIDs[j], Next: s.deadNext[j], Sim: float64(scores[i])}
	}
	return hits, nil
}

// paraCosTop: GPU twin of paraphrase.go paraCosTop (present-time only).
func (s *vecStore) paraCosTop(qv []float32, n int) (map[int64]float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids, scores, err := s.topk(vecSetPara, s.paraIDs, qv, n)
	if err != nil {
		return nil, err
	}
	m := make(map[int64]float64, len(ids))
	for i, id := range ids {
		m[id] = float64(scores[i])
	}
	return m, nil
}
