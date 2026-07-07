package ask

// askab: the local-ask gate. Samples N answerable probes from a mined TSV
// (question<TAB>regex<TAB>miner<TAB>as-of; see mineprobes.go) and runs classic
// ask (the frontier LLM tool loop) vs askLocal (local policy + local synth) on each.
// Reports per-arm regex-hit rate (does the probe regex (?i) match any CITED
// fact's statement), wall-clock p50, and frontier the frontier LLM call counts (the
// zero-frontier confirmation for the local arm).
//
// Gate (2026-07-07 scope): local >= 80% of classic's hit rate at p50 < 10s.

import (
	"bufio"
	"database/sql"
	"fmt"
	"math/rand"
	"oracle/internal/llm"
	"oracle/internal/search"
	"oracle/internal/store"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

type abProbe struct {
	q, pattern string
}

type abArm struct {
	name     string
	fn       func(*sql.DB, string, string, float64) (string, []search.FactOut, AskConfidence, error)
	hits     int
	errs     int
	secs     []float64
	frontier int64
}

func RunAskAB(db *sql.DB, probesPath string, n int) error {
	f, err := os.Open(probesPath)
	if err != nil {
		return err
	}
	defer f.Close()
	var probes []abProbe
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		parts := strings.Split(sc.Text(), "\t")
		if len(parts) < 3 {
			continue
		}
		// as-of probes need historical dispatch; the gate is about the
		// current-knowledge path, so skip them.
		if len(parts) > 3 && strings.TrimSpace(parts[3]) != "" {
			continue
		}
		if _, err := regexp.Compile("(?i)" + parts[1]); err != nil {
			continue
		}
		probes = append(probes, abProbe{q: parts[0], pattern: parts[1]})
	}
	rng := rand.New(rand.NewSource(42)) // deterministic sample
	rng.Shuffle(len(probes), func(i, j int) { probes[i], probes[j] = probes[j], probes[i] })
	if len(probes) > n {
		probes = probes[:n]
	}

	arms := []*abArm{
		{name: "classic", fn: ask},
		{name: "local", fn: askLocal},
	}
	for i, p := range probes {
		re := regexp.MustCompile("(?i)" + p.pattern)
		for _, arm := range arms {
			before := llm.FrontierCalls.Load()
			t0 := time.Now()
			answer, facts, _, err := arm.fn(db, p.q, "", 0)
			dt := time.Since(t0).Seconds()
			arm.frontier += llm.FrontierCalls.Load() - before
			if err != nil {
				arm.errs++
				fmt.Printf("[%d/%d] %-7s ERR %.1fs %v\n", i+1, len(probes), arm.name, dt, err)
				continue
			}
			arm.secs = append(arm.secs, dt)
			// regex-hit proxy: pattern matches a CITED fact's statement (the
			// probes were mined from statements), or the answer text itself.
			seen := map[int64]search.FactOut{}
			for _, h := range facts {
				seen[h.ID] = h
			}
			hit := re.MatchString(answer)
			for _, id := range citedIDs(answer, seen) {
				if re.MatchString(seen[id].Statement) {
					hit = true
				}
			}
			if hit {
				arm.hits++
			}
			fmt.Printf("[%d/%d] %-7s hit=%-5v %.1fs %s\n", i+1, len(probes), arm.name, hit, dt, store.Truncate(p.q, 70))
		}
	}

	fmt.Println("\n=== ask A/B gate ===")
	for _, arm := range arms {
		fmt.Printf("%-7s hit %d/%d (%.1f%%)  p50 %.1fs  errors %d  the frontier LLM calls %d\n",
			arm.name, arm.hits, len(probes), 100*float64(arm.hits)/float64(len(probes)),
			p50(arm.secs), arm.errs, arm.frontier)
	}
	return nil
}

func p50(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}
