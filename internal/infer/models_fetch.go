package infer

// Self-provisioning model weights: download + unpack the models-v1 release
// assets into ~/.oracle/models. Used by `oracle models pull` and lazily at
// inference init when a local-model flag is on but weights are absent.
// Auth: GITHUB_TOKEN env, else the gh CLI's stored oauth token.

import (
	"crypto/sha256"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"oracle/internal/store"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// The manifest pins weights to the code: models/manifest.json lives in git,
// names the release tag and per-asset sha256. Old commits fetch their own tag.
type modelManifest struct {
	ReleaseTag string `json:"release_tag"`
	Repo       string `json:"repo"`
	Assets     map[string]struct {
		File   string `json:"file"`
		Sha256 string `json:"sha256"`
	} `json:"assets"`
}

//go:embed models/manifest.json
var manifestBytes []byte

func LoadManifest() (modelManifest, error) {
	var m modelManifest
	err := json.Unmarshal(manifestBytes, &m)
	if err == nil && (m.ReleaseTag == "" || len(m.Assets) == 0) {
		err = fmt.Errorf("manifest empty")
	}
	return m, err
}

func githubToken() (string, error) {
	if t := os.Getenv("GITHUB_TOKEN"); t != "" {
		return t, nil
	}
	out, err := exec.Command("gh", "auth", "token").Output()
	if err != nil {
		return "", fmt.Errorf("model weights live on a private GitHub release — run `gh auth login` once, or set GITHUB_TOKEN to a PAT with repo read access (no GITHUB_TOKEN set and `gh auth token` failed: %w)", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// platformAssets filters the manifest to what this platform can use — the
// MLX Metal bundle ("lib") is dead weight off-darwin. Sorted for stable
// output.
func PlatformAssets(m modelManifest) []string {
	dirs := make([]string, 0, len(m.Assets))
	for d := range m.Assets {
		if d == "lib" && runtime.GOOS != "darwin" {
			continue
		}
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)
	return dirs
}

func ModelsDir() string { return filepath.Join(store.OracleHome(), "models") }

// ensureModels checks the needed dirs exist, pulling missing ones. Loud.
func ensureModels(needed ...string) error {
	var missing []string
	for _, d := range needed {
		if _, err := os.Stat(filepath.Join(ModelsDir(), d)); err != nil {
			missing = append(missing, d)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	m, err := LoadManifest()
	if err != nil {
		return fmt.Errorf("models manifest: %w", err)
	}
	fmt.Fprintf(os.Stderr, "oracle: fetching model weights %v from %s@%s\n", missing, m.Repo, m.ReleaseTag)
	return PullModels(m, missing)
}

func PullModels(m modelManifest, dirs []string) error {
	tok, err := githubToken()
	if err != nil {
		return err
	}
	cl := &http.Client{Timeout: 30 * time.Minute}
	req, _ := http.NewRequest("GET",
		"https://api.github.com/repos/"+m.Repo+"/releases/tags/"+m.ReleaseTag, nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := cl.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var rel struct {
		Assets []struct {
			Name string `json:"name"`
			URL  string `json:"url"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return err
	}
	byName := map[string]string{}
	for _, a := range rel.Assets {
		byName[a.Name] = a.URL
	}
	if err := os.MkdirAll(ModelsDir(), 0o755); err != nil {
		return err
	}
	for _, d := range dirs {
		spec, ok := m.Assets[d]
		if !ok {
			return fmt.Errorf("manifest has no entry for %s", d)
		}
		url, ok := byName[spec.File]
		if !ok {
			return fmt.Errorf("release %s has no asset %s", m.ReleaseTag, spec.File)
		}
		tgz := filepath.Join(ModelsDir(), spec.File)
		if err := downloadAsset(cl, tok, url, tgz); err != nil {
			return fmt.Errorf("download %s: %w", spec.File, err)
		}
		got, err := fileSha256(tgz)
		if err != nil {
			return err
		}
		if got != spec.Sha256 {
			return fmt.Errorf("%s checksum mismatch: got %s want %s — refusing to unpack", spec.File, got[:12], spec.Sha256[:12])
		}
		cmd := exec.Command("tar", "xzf", tgz, "-C", ModelsDir())
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("untar %s: %v: %s", spec.File, err, out)
		}
		fmt.Fprintf(os.Stderr, "oracle: %s ready\n", d)
	}
	return nil
}

// downloadAsset streams url to dest (atomic via .part). Empty tok = public
// download, no auth header.
func downloadAsset(cl *http.Client, tok, url, dest string) error {
	req, _ := http.NewRequest("GET", url, nil)
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	req.Header.Set("Accept", "application/octet-stream")
	resp, err := cl.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, b)
	}
	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(tmp, dest)
}

func fileSha256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func ModelsStatus() {
	m, err := LoadManifest()
	if err != nil {
		fmt.Println("manifest error:", err)
		return
	}
	fmt.Printf("pinned release: %s\n", m.ReleaseTag)
	names := make([]string, 0, len(m.Assets))
	for d := range m.Assets {
		names = append(names, d)
	}
	sort.Strings(names)
	for _, d := range names {
		if d == "lib" && runtime.GOOS != "darwin" {
			fmt.Printf("  %-18s (mlx, darwin-only — skipped here)\n", d)
			continue
		}
		p := filepath.Join(ModelsDir(), d)
		if fi, err := os.Stat(p); err == nil && fi.IsDir() {
			fmt.Printf("  %-18s present\n", d)
		} else {
			fmt.Printf("  %-18s MISSING (oracle models pull)\n", d)
		}
	}
	if p, err := findORTDylib(); err == nil {
		fmt.Printf("  %-18s present (%s)\n", "onnxruntime", p)
	} else {
		fmt.Printf("  %-18s MISSING (oracle models pull fetches v%s)\n", "onnxruntime", ortVersion)
	}
}

// --- ONNX Runtime self-provisioning ---
//
// The local judge/embedder need libonnxruntime. A system install (brew, apt,
// /usr/local/lib) is respected; when none is found, `oracle models pull`
// fetches the pinned official release — public repo, no token — into
// ~/.oracle/lib, which findORTDylib checks first after ORT_DYLIB.

// ortVersion is pinned: onnxruntime_go v1.31.0 requires ORT C API 26, which
// shipped in 1.26.x (1.22/1.23 fail with "requested API version not
// available").
const ortVersion = "1.26.0"

// ortAssets pins the official release asset + sha256 per platform (hashes
// computed from the v1.26.0 release assets, 2026-07-07). No osx-x86_64:
// upstream stopped shipping it — Intel Macs use brew.
var ortAssets = map[string]struct{ file, sha256 string }{
	"linux-amd64":  {"onnxruntime-linux-x64-" + ortVersion + ".tgz", "1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57"},
	"linux-arm64":  {"onnxruntime-linux-aarch64-" + ortVersion + ".tgz", "34ff1c2d0f12e2cf3d33a0c5f82e39792e1d581fbd6968fd7c30d173654be01a"},
	"darwin-arm64": {"onnxruntime-osx-arm64-" + ortVersion + ".tgz", "7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3"},
}

// ortProvisionedDylib is where a pulled runtime's library lands ("" =
// platform without a pinned build).
func ortProvisionedDylib() string {
	a, ok := ortAssets[runtime.GOOS+"-"+runtime.GOARCH]
	if !ok {
		return ""
	}
	name := "libonnxruntime.so"
	if runtime.GOOS == "darwin" {
		name = "libonnxruntime.dylib"
	}
	return filepath.Join(store.OracleHome(), "lib", strings.TrimSuffix(a.file, ".tgz"), "lib", name)
}

// ensureORT provisions libonnxruntime if no usable copy exists. Checksum
// failure refuses to unpack, same policy as model weights.
func EnsureORT() error {
	if _, err := findORTDylib(); err == nil {
		return nil // system or previously provisioned copy wins
	}
	key := runtime.GOOS + "-" + runtime.GOARCH
	a, ok := ortAssets[key]
	if !ok {
		return fmt.Errorf("models: no pinned onnxruntime build for %s — install libonnxruntime (brew/apt) or set ORT_DYLIB", key)
	}
	libDir := filepath.Join(store.OracleHome(), "lib")
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		return err
	}
	url := "https://github.com/microsoft/onnxruntime/releases/download/v" + ortVersion + "/" + a.file
	fmt.Fprintf(os.Stderr, "oracle: fetching onnxruntime %s (%s)\n", ortVersion, key)
	cl := &http.Client{Timeout: 30 * time.Minute}
	tgz := filepath.Join(libDir, a.file)
	if err := downloadAsset(cl, "", url, tgz); err != nil {
		return fmt.Errorf("download onnxruntime: %w", err)
	}
	got, err := fileSha256(tgz)
	if err != nil {
		return err
	}
	if got != a.sha256 {
		return fmt.Errorf("%s checksum mismatch: got %s want %s — refusing to unpack", a.file, got[:12], a.sha256[:12])
	}
	if out, err := exec.Command("tar", "xzf", tgz, "-C", libDir).CombinedOutput(); err != nil {
		return fmt.Errorf("untar %s: %v: %s", a.file, err, out)
	}
	p := ortProvisionedDylib()
	if _, err := os.Stat(p); err != nil {
		return fmt.Errorf("onnxruntime unpacked but %s is missing: %w", p, err)
	}
	fmt.Fprintf(os.Stderr, "oracle: onnxruntime ready (%s)\n", p)
	return nil
}
