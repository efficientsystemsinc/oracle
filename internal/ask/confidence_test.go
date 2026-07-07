package ask

// Tests for confidence.go.

import (
	"oracle/internal/search"
	"strings"
	"testing"
)

func TestAskConfidence(t *testing.T) {
	seen := map[int64]search.FactOut{
		1: {ID: 1, Confidence: 0.9, Evidence: "verified"},
		2: {ID: 2, Confidence: 0.7, Evidence: "asserted"},
		3: {ID: 3, Confidence: 0.8, Evidence: "reported", ContraBy: []int64{99}},
	}
	// strong answer: 3 valid citations, healthy searches
	strong := askConfidence("a [1] b [2] c [3]", []int64{1, 2, 3}, seen, []float64{0.035, 0.03})
	if strong.Score < AskAbstainThreshold || strong.Abstained {
		t.Errorf("strong answer should clear threshold, got %+v", strong)
	}
	if strong.Features.Coverage != 3 || strong.Features.CitationValidity != 1.0 {
		t.Errorf("bad features: %+v", strong.Features)
	}
	if strong.Features.Agreement <= 0.6 || strong.Features.Agreement >= 0.7 { // 2/3
		t.Errorf("agreement want 2/3, got %v", strong.Features.Agreement)
	}

	// uncited answer: everything but retrieval is 0 => must abstain
	weak := askConfidence("not covered in the graph", nil, seen, []float64{0.03})
	if !weak.Abstained || weak.Score > 0.11 {
		t.Errorf("uncited answer must abstain, got %+v", weak)
	}

	// invalid citations dilute validity
	mixed := askConfidence("x [1] y [777] z [888]", []int64{1}, seen, nil)
	if v := mixed.Features.CitationValidity; v < 0.33 || v > 0.34 {
		t.Errorf("validity want 1/3, got %v", v)
	}

	// monotone: more coverage never lowers the score
	one := askConfidence("a [1]", []int64{1}, seen, []float64{0.03})
	two := askConfidence("a [1] b [2]", []int64{1, 2}, seen, []float64{0.03})
	if two.Score < one.Score {
		t.Errorf("coverage must be monotone: %v -> %v", one.Score, two.Score)
	}
}

func TestApplyAbstain(t *testing.T) {
	c := AskConfidence{Score: 0.1, Abstained: true}
	out := applyAbstain("nothing in the graph about X.", c)
	if !strings.HasPrefix(out, abstainPrefix) {
		t.Errorf("abstained answer must lead with marker, got %q", out)
	}
	// idempotent when the reasoner already leads with it
	if got := applyAbstain(out, c); strings.Count(got, abstainPrefix) != 1 {
		t.Errorf("marker duplicated: %q", got)
	}
	if got := applyAbstain("fine [1]", AskConfidence{Score: 0.9}); got != "fine [1]" {
		t.Errorf("confident answer must be untouched, got %q", got)
	}
}

func TestLeadsWithDoubt(t *testing.T) {
	for _, s := range []string{
		"We don’t appear to use an Elasticsearch cluster for search. [1]",
		"Unknown — I don't have a cited fact stating the warehouse size.",
		"We **did not have a recorded decision** to build it [2].",
		abstainPrefix + " nothing in the graph about Solana.",
	} {
		if !leadsWithDoubt(s) {
			t.Errorf("want doubt lead: %q", s)
		}
	}
	for _, s := range []string{
		"pg_pool_min_size = 4 [1118], pg_pool_max_size = 32 [29621]. these are the current defaults per the enriched settings facts for the quasar pg pool configuration. caveat: an older fact found no override.",
		"atlas01 runs driver 580.159.03 [12].",
	} {
		if leadsWithDoubt(s) {
			t.Errorf("false doubt on confident lead: %q", s)
		}
	}
}

func TestDoubtDampsScore(t *testing.T) {
	seen := map[int64]search.FactOut{1: {ID: 1, Confidence: 0.9, Evidence: "verified"}, 2: {ID: 2, Confidence: 0.9, Evidence: "verified"}, 3: {ID: 3, Confidence: 0.9, Evidence: "verified"}}
	confident := askConfidence("we use X [1][2][3]", []int64{1, 2, 3}, seen, []float64{0.03})
	negative := askConfidence("we don't use Elasticsearch; search is X [1][2][3]", []int64{1, 2, 3}, seen, []float64{0.03})
	if !negative.Abstained || negative.Score >= confident.Score*0.5 {
		t.Errorf("doubt lead must damp below threshold: confident=%v negative=%v", confident.Score, negative.Score)
	}
}
