package search

// Self-labeling eval — no authored probes. Ground truth comes from the graph's
// own structure and from real usage, so it grows with the DB and cannot be
// saturated by probe authoring:
//   - supersession pairs: querying the old statement AT ITS ERA (--as-of the
//     midpoint between old and new valid_from) must retrieve the old fact;
//     querying it TODAY must retrieve the superseder (continuity under drift).
//   - citation replay: for real `ask` traces, plain search on the original
//     question should recall the facts the reasoner ultimately cited.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"oracle/internal/store"
)

type SelfEval struct {
	Pairs          int     `json:"pairs"`
	EraHit         int     `json:"era_hit"`
	NowHit         int     `json:"now_hit"`
	AskTraces      int     `json:"ask_traces"`
	CitationRecall float64 `json:"citation_recall"`
	K              int     `json:"k"`
}

type supersessionPair struct {
	oldID, newID       int64
	oldStmt            string
	oldValid, newValid float64
}

// supersessionPairs returns the newest n supersessions whose old/new facts are
// separated by world time (>1h), so the as-of midpoint is meaningful.
func supersessionPairs(db *sql.DB, n int) ([]supersessionPair, error) {
	rows, err := db.Query(`SELECT f.id, f.statement, f.valid_from, s.id, s.valid_from
		FROM facts f JOIN facts s ON s.id = f.superseded_by
		WHERE s.valid_from > f.valid_from + 3600
		ORDER BY f.id DESC LIMIT ?`, n)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []supersessionPair
	for rows.Next() {
		var p supersessionPair
		if err := rows.Scan(&p.oldID, &p.oldStmt, &p.oldValid, &p.newID, &p.newValid); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

// chainHead walks superseded_by links to the live end of a chain.
func chainHead(db *sql.DB, id int64) int64 {
	head := id
	for hops := 0; hops < 20; hops++ {
		var next sql.NullInt64
		err := db.QueryRow("SELECT superseded_by FROM facts WHERE id = ? AND superseded_at IS NOT NULL", head).Scan(&next)
		if err != nil || !next.Valid || next.Int64 == 0 {
			return head
		}
		head = next.Int64
	}
	return head
}

func containsFact(hits []FactOut, id int64) bool {
	for _, h := range hits {
		if h.ID == id {
			return true
		}
	}
	return false
}

func RunSelfEval(db *sql.DB, sample, k int, verbose bool) (SelfEval, error) {
	ev := SelfEval{K: k}
	pairs, err := supersessionPairs(db, sample)
	if err != nil {
		return ev, err
	}
	for _, p := range pairs {
		mid := (p.oldValid + p.newValid) / 2
		era, err := Search(db, p.oldStmt, "", k, mid, false)
		if err != nil {
			return ev, err
		}
		if containsFact(era, p.oldID) {
			ev.EraHit++
		} else if verbose {
			fmt.Printf("ERA MISS  old=%d  %s\n", p.oldID, store.Truncate(p.oldStmt, 100))
		}
		now, err := Search(db, p.oldStmt, "", k, 0, false)
		if err != nil {
			return ev, err
		}
		// the correct present-day answer is the LIVE HEAD of the chain — the
		// direct superseder may itself be history by now
		head := chainHead(db, p.newID)
		if containsFact(now, head) {
			ev.NowHit++
		} else if verbose {
			fmt.Printf("NOW MISS  old=%d head=%d  %s\n", p.oldID, head, store.Truncate(p.oldStmt, 100))
		}
		ev.Pairs++
	}

	rows, err := db.Query(`SELECT q, results FROM traces WHERE kind='ask' ORDER BY id DESC LIMIT 200`)
	if err != nil {
		return ev, err
	}
	defer rows.Close()
	type askRow struct {
		q     string
		cited []int64
	}
	var asks []askRow
	for rows.Next() {
		var q, res string
		if err := rows.Scan(&q, &res); err != nil {
			return ev, err
		}
		var payload struct {
			Cited []int64 `json:"cited"`
		}
		if json.Unmarshal([]byte(res), &payload) != nil || len(payload.Cited) == 0 {
			continue // pre-instrumentation trace, or answer cited nothing
		}
		asks = append(asks, askRow{q, payload.Cited})
	}
	var recallSum float64
	for _, a := range asks {
		hits, err := Search(db, a.q, "", k, 0, false)
		if err != nil {
			return ev, err
		}
		got := map[int64]bool{}
		for _, h := range hits {
			got[h.ID] = true
		}
		found := 0
		for _, c := range a.cited {
			if got[c] {
				found++
			}
		}
		recallSum += float64(found) / float64(len(a.cited))
		ev.AskTraces++
	}
	if ev.AskTraces > 0 {
		ev.CitationRecall = math.Round(recallSum/float64(ev.AskTraces)*1000) / 1000
	}
	return ev, nil
}

func PrintSelfEval(ev SelfEval) {
	fmt.Printf("supersession pairs: %d\n", ev.Pairs)
	if ev.Pairs > 0 {
		fmt.Printf("  era hit@%d:        %d/%d  (old fact retrieved at its own era via as-of)\n", ev.K, ev.EraHit, ev.Pairs)
		fmt.Printf("  continuity hit@%d: %d/%d  (superseder retrieved today via the old phrasing)\n", ev.K, ev.NowHit, ev.Pairs)
	}
	if ev.AskTraces > 0 {
		fmt.Printf("ask citation replay: mean recall@%d %.3f over %d traces\n", ev.K, ev.CitationRecall, ev.AskTraces)
	} else {
		fmt.Println("ask citation replay: no cited ask traces yet (accumulates with usage)")
	}
}
