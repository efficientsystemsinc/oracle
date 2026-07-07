package ask

// Tests for asklocal.go.

import "testing"

func TestParsePolicyPlan(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []policyStep
	}{
		{"basic", "search(meadow ssh)\nentity(meadow)\nSTOP", []policyStep{
			{"search", "meadow ssh"}, {"entity", "meadow"}}},
		{"stop-parens-and-dupes", "search(x)\nsearch(x)\nSTOP()\nSTOP\nsearch(after stop)", []policyStep{
			{"search", "x"}}},
		{"junk-lines-skipped", "hmm\nsearch(q one)\nnope()\ngraph(quasar)\nmetric(recall_at_10)", []policyStep{
			{"search", "q one"}, {"graph", "quasar"}, {"metric", "recall_at_10"}}},
		{"cap-4", "search(a)\nsearch(b)\nsearch(c)\nsearch(d)\nsearch(e)", []policyStep{
			{"search", "a"}, {"search", "b"}, {"search", "c"}, {"search", "d"}}},
		{"quoted-arg", `search("pgbouncer dsn")`, []policyStep{{"search", "pgbouncer dsn"}}},
		{"empty", "STOP\n", nil},
	}
	for _, c := range cases {
		got := parsePolicyPlan(c.raw)
		if len(got) != len(c.want) {
			t.Fatalf("%s: got %v want %v", c.name, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%s[%d]: got %v want %v", c.name, i, got[i], c.want[i])
			}
		}
	}
}

func TestRetrievalStrength(t *testing.T) {
	if v := retrievalStrength(nil); v != 0 {
		t.Fatalf("empty: %v", v)
	}
	if v := retrievalStrength([]float64{0.03, 0.03}); v != 1 {
		t.Fatalf("strong: %v", v)
	}
	if v := retrievalStrength([]float64{0, 0.015}); v != 0.25 {
		t.Fatalf("weak: %v", v)
	}
}
