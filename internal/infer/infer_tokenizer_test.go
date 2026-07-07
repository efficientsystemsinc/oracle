package infer

// Golden cases generated ONCE with python transformers 5.8.1:
//   BertTokenizerFast.from_pretrained("~/.oracle/models/judge_v2_onnx")
//   t(a, b, truncation=True, max_length=ml, padding="max_length")

import (
	"path/filepath"
	"strings"
	"testing"
)

func loadTestTokenizer(t testing.TB) *wordPieceTokenizer {
	t.Helper()
	p := filepath.Join(oracleModelDir("judge_v2_onnx"), "vocab.txt")
	tok, err := loadWordPieceTokenizer(p)
	if err != nil {
		t.Skipf("judge vocab not available: %v", err)
	}
	return tok
}

type goldenCase struct {
	name   string
	a, b   string
	maxLen int
	ids    []int64
	tt     []int64
	am     []int64
}

func goldenCases() []goldenCase {
	return []goldenCase{
		{
			name: "simple", a: "hello world", maxLen: 16,
			ids: []int64{101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			tt:  []int64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			am:  []int64{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "mixed_case_punct_digits", a: "The quick BROWN fox jumps over 123 lazy dogs!", maxLen: 32,
			ids: []int64{101, 1996, 4248, 2829, 4419, 14523, 2058, 13138, 13971, 6077, 999, 102,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			tt: make([]int64, 32),
			am: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "accents_dash", a: "café naïve résumé — über", maxLen: 32,
			ids: []int64{101, 7668, 15743, 13746, 1517, 19169, 102,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			tt: make([]int64, 32),
			am: []int64{1, 1, 1, 1, 1, 1, 1,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "emoji_unk", a: "I love pizza \U0001F355\U0001F389 and sushi", maxLen: 32,
			ids: []int64{101, 1045, 2293, 10733, 100, 1998, 10514, 6182, 102,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			tt: make([]int64, 32),
			am: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "pair_truncated", a: "deployment failed on atlas01",
			b: "the H100 box quasar-prod-flex-us-east4 is serving traffic", maxLen: 24,
			ids: []int64{101, 10813, 3478, 2006, 11568, 24096, 102, 1996, 1044, 18613, 3482,
				24209, 16782, 2099, 1011, 4013, 2094, 1011, 23951, 1011, 2149, 1011, 2264, 102},
			tt: []int64{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			am: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name: "long_truncation", a: strings.Repeat("word ", 100), maxLen: 16,
			ids: []int64{101, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 2773, 102},
			tt:  make([]int64, 16),
			am:  []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
	}
}

func eqI64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestWordPieceGolden(t *testing.T) {
	tok := loadTestTokenizer(t)
	for _, c := range goldenCases() {
		t.Run(c.name, func(t *testing.T) {
			ids, am, tt := tok.Encode(c.a, c.b, c.maxLen)
			if !eqI64(ids, c.ids) {
				t.Errorf("input_ids\n got %v\nwant %v", ids, c.ids)
			}
			if !eqI64(tt, c.tt) {
				t.Errorf("token_type_ids\n got %v\nwant %v", tt, c.tt)
			}
			if !eqI64(am, c.am) {
				t.Errorf("attention_mask\n got %v\nwant %v", am, c.am)
			}
		})
	}
}
