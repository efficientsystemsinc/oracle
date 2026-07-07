package infer

// infer_mlx_test.go — PARITY GATE (MLX vs ONNX Runtime) + MLX benchmarks.
//
// Skips with a NOTICE when the MLX dylib / exported weights / models are
// absent (e.g. linux CI). When it runs, failures are loud: embed cosine must
// be >= 0.999 per vector and judge argmax identical with max prob diff < 5e-2
// (fp16 tolerance).

import (
	"math"
	"testing"
)

var parityInputs = [8]string{
	benchOld,
	benchNew,
	"query: where is the prod db",
	"passage: prod is postgres at 192.0.2.10, fronted by quasar-pgbouncer",
	"the deploy script clobbers uncommitted box hotfixes; diff the box against main first",
	"oracle is a Go daemon on :4141 with a bi-temporal fact graph over agent sessions",
	"meadow SSH is always user meadow-user; .101 uses meadow_id_ed25519",
	"short",
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestMLXEmbedParity(t *testing.T) {
	RequireMLX(t)
	requireEmbed(t)
	t.Setenv("ORACLE_MLX", "") // force the ORT reference path in embedLocal
	texts := parityInputs[:]
	ref, err := EmbedLocal(texts)
	if err != nil {
		t.Fatalf("ONNX reference failed: %v", err)
	}
	got, err := embedMLX(texts)
	if err != nil {
		t.Fatalf("MLX embed failed: %v", err)
	}
	for i := range texts {
		if len(got[i]) != 512 {
			t.Fatalf("input %d: MLX dim %d, want 512", i, len(got[i]))
		}
		c := cosine(ref[i], got[i])
		t.Logf("input %d: cosine(onnx, mlx) = %.6f", i, c)
		if c < 0.999 {
			t.Errorf("PARITY FAILURE input %d (%q): cosine %.6f < 0.999", i, texts[i], c)
		}
	}
}

func TestMLXJudgeParity(t *testing.T) {
	RequireMLX(t)
	requireJudge(t)
	t.Setenv("ORACLE_MLX", "") // force the ORT reference path in judgeLocal
	for i := 0; i < len(parityInputs); i++ {
		a := parityInputs[i]
		b := parityInputs[(i+3)%len(parityInputs)]
		ref, err := JudgeLocal(a, b)
		if err != nil {
			t.Fatalf("ONNX reference failed: %v", err)
		}
		got, err := judgeMLX(a, b)
		if err != nil {
			t.Fatalf("MLX judge failed: %v", err)
		}
		refArg, gotArg, maxDiff := 0, 0, 0.0
		for j := 1; j < 3; j++ {
			if ref[j] > ref[refArg] {
				refArg = j
			}
			if got[j] > got[gotArg] {
				gotArg = j
			}
		}
		for j := 0; j < 3; j++ {
			if d := math.Abs(ref[j] - got[j]); d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("pair %d: onnx=%v mlx=%v maxdiff=%.4f", i, ref, got, maxDiff)
		if refArg != gotArg {
			t.Errorf("PARITY FAILURE pair %d: argmax onnx=%d mlx=%d (onnx=%v mlx=%v)",
				i, refArg, gotArg, ref, got)
		}
		if maxDiff >= 5e-2 {
			t.Errorf("PARITY FAILURE pair %d: max prob diff %.4f >= 0.05", i, maxDiff)
		}
	}
}

// --- benchmarks (mirror the ORT ones in infer_onnx_test.go) ---

func BenchmarkMLXJudgeSingle(b *testing.B) {
	RequireMLX(b)
	if _, err := judgeMLX(benchOld, benchNew); err != nil { // warm (compile cache)
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := judgeMLX(benchOld, benchNew); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLXJudgeBatch16(b *testing.B) {
	RequireMLX(b)
	a := make([]string, 16)
	pair := make([]string, 16)
	for i := range a {
		a[i], pair[i] = benchOld, benchNew
	}
	if _, err := judgeMLXBatch(a, pair); err != nil { // warm at batch shape
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := judgeMLXBatch(a, pair); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLXEmbedSingle(b *testing.B) {
	RequireMLX(b)
	texts := []string{benchNew}
	if _, err := embedMLX(texts); err != nil { // warm
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := embedMLX(texts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLXEmbedBatch16(b *testing.B) {
	RequireMLX(b)
	texts := make([]string, 16)
	for i := range texts {
		texts[i] = benchNew
	}
	if _, err := embedMLX(texts); err != nil { // warm at batch shape
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := embedMLX(texts); err != nil {
			b.Fatal(err)
		}
	}
}
