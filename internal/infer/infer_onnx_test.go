package infer

// Tests for infer_onnx.go.

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	benchOld = "the prod API box is quasar-h100-flex-mig-62gk at 34.60.15.89, reached over IAP; the old 34.41.54.123 box is gone"
	benchNew = "quasar.computer now resolves to 35.234.178.86 (quasar-prod-flex-us-east4, us-east4-a a3-highgpu-8g); /v1/health reports planner_model=quasar-v8 and storage ok"
)

func requireJudge(t testing.TB) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(oracleModelDir("judge_v2_onnx"), "model.onnx")); err != nil {
		t.Skipf("judge model not available: %v", err)
	}
	t.Setenv("ORACLE_LOCAL_JUDGE", "1")
}

func requireEmbed(t testing.TB) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(oracleModelDir("embedder_v3_onnx"), "model.onnx")); err != nil {
		t.Skipf("embedder model not available: %v", err)
	}
	t.Setenv("ORACLE_LOCAL_EMBED", "1")
}

func TestJudgeLocalFlagOff(t *testing.T) {
	t.Setenv("ORACLE_LOCAL_JUDGE", "")
	if _, err := JudgeLocal("a", "b"); err == nil {
		t.Fatal("judgeLocal with flag off should error")
	}
	t.Setenv("ORACLE_LOCAL_EMBED", "")
	if _, err := EmbedLocal([]string{"a"}); err == nil {
		t.Fatal("embedLocal with flag off should error")
	}
}

func TestJudgeLocal(t *testing.T) {
	requireJudge(t)
	probs, err := JudgeLocal(benchOld, benchNew)
	if err != nil {
		t.Fatal(err)
	}
	sum := 0.0
	for _, p := range probs {
		if p < 0 || p > 1 || math.IsNaN(p) {
			t.Fatalf("bad prob %v", probs)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("probs sum %v != 1 (%v)", sum, probs)
	}
	t.Logf("probs = %v", probs)
}

func TestEmbedLocal(t *testing.T) {
	requireEmbed(t)
	vecs, err := EmbedLocal([]string{"query: where is the prod db", "passage: prod is postgres at 192.0.2.10"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 512 {
		t.Fatalf("shape: %d x %d", len(vecs), len(vecs[0]))
	}
	var norm float64
	for _, v := range vecs[0] {
		norm += float64(v) * float64(v)
	}
	if norm == 0 || math.IsNaN(norm) {
		t.Fatalf("degenerate embedding, |v|^2=%v", norm)
	}
}

func BenchmarkJudgeSingle(b *testing.B) {
	requireJudge(b)
	if _, err := JudgeLocal(benchOld, benchNew); err != nil { // warm
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := JudgeLocal(benchOld, benchNew); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJudgeBatch16(b *testing.B) {
	requireJudge(b)
	if _, err := JudgeLocal(benchOld, benchNew); err != nil { // warm + init
		b.Fatal(err)
	}
	a := make([]string, 16)
	pair := make([]string, 16)
	for i := range a {
		a[i], pair[i] = benchOld, benchNew
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := judgeModel.run(a, pair, judgeMaxLen); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedSingle(b *testing.B) {
	requireEmbed(b)
	texts := []string{benchNew}
	if _, err := EmbedLocal(texts); err != nil { // warm
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EmbedLocal(texts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedBatch16(b *testing.B) {
	requireEmbed(b)
	texts := make([]string, 16)
	for i := range texts {
		texts[i] = benchNew
	}
	if _, err := EmbedLocal(texts[:1]); err != nil { // warm
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EmbedLocal(texts); err != nil {
			b.Fatal(err)
		}
	}
}
