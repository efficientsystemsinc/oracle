package ingest

// `oracle canary` — extraction canaries. Runs the real LLM extraction over
// three fixed transcript fixtures (testdata/canary_chunks/*.jsonl, claude-code
// format) and checks the output against testdata/canary_expected.json using
// loose keyword matching: each expected fact is a keyword set, and it passes
// when at least one extracted statement contains ALL its keywords
// (case-insensitive). Catches prompt / model / parsing regressions without
// pinning exact phrasing.
//
// `oracle canary --gen` prints the extracted statements per chunk instead of
// judging, to help (re)author canary_expected.json by hand.

import (
	"encoding/json"
	"fmt"
	"oracle/internal/store"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// canaryExpected: chunk filename -> list of expected facts, each a keyword set.
type canaryExpected map[string][][]string

func RunCanary(dir, expectedPath string, gen bool) error {
	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fmt.Errorf("no canary chunks in %s", dir)
	}
	sort.Strings(files)

	var expected canaryExpected
	if !gen {
		b, err := os.ReadFile(expectedPath)
		if err != nil {
			return fmt.Errorf("read expected: %w", err)
		}
		if err := json.Unmarshal(b, &expected); err != nil {
			return fmt.Errorf("parse %s: %w", expectedPath, err)
		}
	}

	failed := 0
	for _, f := range files {
		name := filepath.Base(f)
		stmts, err := canaryExtract(f)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if gen {
			fmt.Printf("== %s (%d facts)\n", name, len(stmts))
			for _, s := range stmts {
				fmt.Println("  -", s)
			}
			continue
		}
		exp, ok := expected[name]
		if !ok {
			return fmt.Errorf("%s: no entry in %s", name, expectedPath)
		}
		missing := 0
		for _, kws := range exp {
			if !anyStatementMatches(stmts, kws) {
				missing++
				fmt.Printf("MISS %s: no extracted statement contains all of %v\n", name, kws)
			}
		}
		if missing > 0 {
			failed++
			fmt.Printf("FAIL %s: %d/%d expected facts missing (extracted %d statements)\n",
				name, missing, len(exp), len(stmts))
			for _, s := range stmts {
				fmt.Println("  got:", store.Truncate(s, 160))
			}
		} else {
			fmt.Printf("PASS %s: %d/%d expected facts found (extracted %d statements)\n",
				name, len(exp), len(exp), len(stmts))
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d/%d canary chunks failed", failed, len(files))
	}
	return nil
}

// canaryExtract renders one fixture through the real transcript reader
// (readNew, claude format) and runs LLM extraction on the joined chunk text.
func canaryExtract(fixture string) ([]string, error) {
	b, err := os.ReadFile(fixture)
	if err != nil {
		return nil, err
	}
	// readNew stats + opens by path; give it a throwaway copy so byte offsets
	// and repo resolution behave exactly as in production.
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("oracle-canary-%d-%s", os.Getpid(), filepath.Base(fixture)))
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return nil, err
	}
	defer os.Remove(tmp)
	chunks, err := ReadNew(tmp, "claude", 0, "")
	if err != nil {
		return nil, err
	}
	var texts []string
	repo, eventTime := "unknown", float64(time.Now().Unix())
	for _, c := range chunks {
		if strings.TrimSpace(c.Text) != "" {
			texts = append(texts, c.Text)
		}
		repo, eventTime = c.Repo, c.EventTime
	}
	if len(texts) == 0 {
		return nil, fmt.Errorf("fixture rendered no transcript text")
	}
	date := store.LocalDate(eventTime)
	facts, err := ExtractFacts(strings.Join(texts, "\n\n"), repo, date)
	if err != nil {
		return nil, err
	}
	var stmts []string
	for _, f := range facts {
		stmts = append(stmts, f.Statement)
	}
	return stmts, nil
}

func anyStatementMatches(stmts []string, keywords []string) bool {
	for _, s := range stmts {
		ls := strings.ToLower(s)
		all := true
		for _, kw := range keywords {
			if !strings.Contains(ls, strings.ToLower(kw)) {
				all = false
				break
			}
		}
		if all {
			return true
		}
	}
	return false
}
