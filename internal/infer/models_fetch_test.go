package infer

// models_fetch_test.go — offline checks on the provisioning helpers: platform
// filtering of manifest assets and the pinned-ORT path shape. No network.

import (
	"runtime"
	"strings"
	"testing"
)

func TestPlatformAssetsSkipsMLXLibOffDarwin(t *testing.T) {
	m := modelManifest{
		ReleaseTag: "models-v1",
		Assets: map[string]struct {
			File   string `json:"file"`
			Sha256 string `json:"sha256"`
		}{
			"judge_v2_onnx":    {File: "judge_v2_onnx.tgz"},
			"embedder_v3_onnx": {File: "embedder_v3_onnx.tgz"},
			"lib":              {File: "mlx_lib.tgz"},
		},
	}
	got := PlatformAssets(m)
	hasLib := false
	for _, d := range got {
		if d == "lib" {
			hasLib = true
		}
	}
	if runtime.GOOS == "darwin" && !hasLib {
		t.Fatal("darwin must include the mlx lib bundle")
	}
	if runtime.GOOS != "darwin" && hasLib {
		t.Fatal("non-darwin must skip the mlx lib bundle")
	}
	want := 2
	if runtime.GOOS == "darwin" {
		want = 3
	}
	if len(got) != want {
		t.Fatalf("got %v, want %d assets", got, want)
	}
}

func TestORTProvisionedDylibShape(t *testing.T) {
	p := ortProvisionedDylib()
	key := runtime.GOOS + "-" + runtime.GOARCH
	if _, pinned := ortAssets[key]; !pinned {
		if p != "" {
			t.Fatalf("unpinned platform %s must yield empty path, got %s", key, p)
		}
		return
	}
	if p == "" {
		t.Fatalf("pinned platform %s must yield a path", key)
	}
	if !strings.Contains(p, ortVersion) || !strings.Contains(p, "libonnxruntime") {
		t.Fatalf("provisioned path looks wrong: %s", p)
	}
}
