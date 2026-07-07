package ask

// askeval: blind confidence/abstain evaluation for ask.
//
// Probes live in a JSON file (default eval/ask_confidence_probes.json):
// 25 answerable questions mined from the fact graph (expect = gold substrings,
// any-match counts as correct) + 15 questions about topics verifiably absent.
// Splits are fixed in the file: threshold is FIT on "build" and REPORTED on
// "holdout" — never tuned on the reported split (house rule).
//
// Run against a COPY of the live DB (ORACLE_HOME=<dir with oracle.db copy>):
// ask reinforces cited facts and logs traces, i.e. it writes.

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"oracle/internal/store"
	"os"
	"sort"
	"strings"
)

type askProbe struct {
	ID         string   `json:"id"`
	Split      string   `json:"split"` // build | holdout
	Answerable bool     `json:"answerable"`
	Q          string   `json:"q"`
	Expect     []string `json:"expect,omitempty"` // lowercase substrings; any match = correct
}

type askEvalRow struct {
	askProbe
	Confidence float64     `json:"confidence"`
	Abstained  bool        `json:"abstained"`
	Correct    bool        `json:"correct"` // answerable: expect matched; unanswerable: abstained
	Features   AskFeatures `json:"features"`
	Answer     string      `json:"answer"`
}

func RunAskEval(db *sql.DB, probesPath, outPath string, threshold float64, split string) error {
	raw, err := os.ReadFile(probesPath)
	if err != nil {
		return err
	}
	var probes []askProbe
	if err := json.Unmarshal(raw, &probes); err != nil {
		return fmt.Errorf("parse %s: %w", probesPath, err)
	}
	var outF *os.File
	var enc *json.Encoder
	if outPath != "" {
		f, err := os.Create(outPath)
		if err != nil {
			return err
		}
		outF = f
		defer outF.Close()
		enc = json.NewEncoder(f)
	}
	var rows []askEvalRow
	for i, p := range probes {
		if split != "" && p.Split != split {
			continue
		}
		answer, _, conf, err := ask(db, p.Q, "", 0)
		if err != nil {
			return fmt.Errorf("probe %s: %w", p.ID, err)
		}
		abst := conf.Score < threshold
		row := askEvalRow{askProbe: p, Confidence: conf.Score, Abstained: abst, Features: conf.Features, Answer: answer}
		if p.Answerable {
			la := strings.ToLower(answer)
			for _, e := range p.Expect {
				if strings.Contains(la, strings.ToLower(e)) {
					row.Correct = true
					break
				}
			}
		} else {
			// unanswerable: abstaining is correct, and so is a GROUNDED NEGATIVE -
			// an answer explicitly stating the topic is absent / not used
			// ("we don't use Elasticsearch; search is X [ids]"). Confidently wrong
			// means asserting a positive fabricated answer above threshold.
			row.Correct = abst || deniesCoverage(answer)
		}
		rows = append(rows, row)
		if enc != nil {
			if err := enc.Encode(row); err != nil {
				return err
			}
		}
		fmt.Printf("[%2d/%d] %-6s %-8s ans=%-5v conf=%.3f abstain=%-5v correct=%v  %s\n",
			i+1, len(probes), p.ID, p.Split, p.Answerable, conf.Score, abst, row.Correct, store.Truncate(p.Q, 60))
	}
	for _, sp := range []string{"build", "holdout"} {
		printAskEvalSummary(rows, sp, threshold)
	}
	printThresholdSweep(rows, "build")
	return nil
}

func printAskEvalSummary(rows []askEvalRow, split string, threshold float64) {
	var ansN, ansAnswered, ansAnsweredCorrect, unN, unAbstained, confWrong int
	for _, r := range rows {
		if r.Split != split {
			continue
		}
		if r.Answerable {
			ansN++
			if !r.Abstained {
				ansAnswered++
				if r.Correct {
					ansAnsweredCorrect++
				} else {
					confWrong++
				}
			}
		} else {
			unN++
			if r.Abstained {
				unAbstained++
			} else {
				confWrong++
			}
		}
	}
	if ansN+unN == 0 {
		return
	}
	fmt.Printf("\n== %s (threshold %.2f) ==\n", split, threshold)
	fmt.Printf("answerable answered:    %d/%d (of answered, correct: %d/%d)\n", ansAnswered, ansN, ansAnsweredCorrect, ansAnswered)
	fmt.Printf("unanswerable abstained: %d/%d\n", unAbstained, unN)
	fmt.Printf("confidently wrong:      %d\n", confWrong)
	fmt.Println("calibration (bucket: correct/n):")
	type bucket struct{ correct, n int }
	buckets := map[int]*bucket{}
	for _, r := range rows {
		if r.Split != split {
			continue
		}
		b := int(r.Confidence * 5)
		if b > 4 {
			b = 4
		}
		if buckets[b] == nil {
			buckets[b] = &bucket{}
		}
		buckets[b].n++
		// calibration target: was the emitted behavior right (answer correct, or
		// abstain on a truly unanswerable question)
		if r.Correct {
			buckets[b].correct++
		}
	}
	var keys []int
	for k := range buckets {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		fmt.Printf("  %.1f-%.1f: %d/%d\n", float64(k)*0.2, float64(k+1)*0.2, buckets[k].correct, buckets[k].n)
	}
}

// printThresholdSweep shows, per candidate threshold, the build-split tradeoff
// used to pick askAbstainThreshold.
func printThresholdSweep(rows []askEvalRow, split string) {
	fmt.Printf("\n== threshold sweep (%s split) ==\n", split)
	fmt.Println("thr   ans-answered  unans-abstained")
	for thr := 0.10; thr < 0.90; thr += 0.05 {
		var ansN, ansA, unN, unA int
		for _, r := range rows {
			if r.Split != split {
				continue
			}
			if r.Answerable {
				ansN++
				if r.Confidence >= thr {
					ansA++
				}
			} else {
				unN++
				if r.Confidence < thr {
					unA++
				}
			}
		}
		if ansN+unN == 0 {
			return
		}
		fmt.Printf("%.2f  %d/%d          %d/%d\n", thr, ansA, ansN, unA, unN)
	}
}

// deniesCoverage reports whether the answer explicitly states the asked-about
// thing is absent, unused, or not in the graph — a grounded negative rather
// than a fabricated positive. Checked only for unanswerable probes.
func deniesCoverage(answer string) bool {
	la := strings.ToLower(strings.ReplaceAll(answer, "\u2019", "'"))
	for _, m := range doubtMarkers {
		if strings.Contains(la, m) {
			return true
		}
	}
	return false
}
