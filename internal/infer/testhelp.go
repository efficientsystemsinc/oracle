package infer

import "testing"

// Test helpers importable by other packages' tests.

func RequireMLX(t testing.TB) {
	t.Helper()
	if ok, reason := mlxAvailable(); !ok {
		t.Skipf("NOTICE: MLX engine unavailable, skipping (ORT path unaffected): %s", reason)
	}
}
