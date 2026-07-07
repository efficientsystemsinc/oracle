package infer

// infer_onnx.go — native ONNX Runtime inference (cgo via github.com/yalue/onnxruntime_go)
// for the two local models, replacing remote API calls:
//
//   ~/.oracle/models/judge_v2_onnx/    bert-base 3-class sequence classifier
//   ~/.oracle/models/embedder_v3_onnx/ e5-base with mean-pool + 512-d projection
//                                      baked into the graph (output "embedding")
//
// ONNX Runtime shared library:
//   macOS:  brew install onnxruntime   (found at /opt/homebrew/lib/libonnxruntime.dylib
//           or /opt/homebrew/opt/onnxruntime/lib/)
//   linux:  install libonnxruntime.so into /usr/lib or /usr/local/lib
//   any OS: ORT_DYLIB=/path/to/libonnxruntime.{dylib,so} overrides discovery.
//
// Default execution is CPU with intra-op threads = 4 (benchmarked faster than
// CoreML for these models); ORACLE_ORT_COREML=1 opts into the CoreML EP on
// darwin. Missing dylib/model with the feature flag on is a loud error
// (ADR-004: no silent fallback).
//
// Feature flags (default off; wiring into call sites is separate work):
//   ORACLE_LOCAL_JUDGE=1  enables judgeLocal
//   ORACLE_LOCAL_EMBED=1  enables embedLocal

import (
	"fmt"
	"math"
	"oracle/internal/store"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	judgeMaxLen = 256
	embedMaxLen = 256
)

func localJudgeEnabled() bool {
	m := os.Getenv("ORACLE_LOCAL_JUDGE")
	return m == "1" || m == "shadow" || m == "active"
}

// mlxEnabled routes embedLocal/judgeLocal to the MLX (Apple Metal) engine in
// cpp/oraclemlx instead of ONNX Runtime. Both flags documented:
//
//	ORACLE_MLX=1        use MLX; missing dylib/weights is a loud error
//	                    (ADR-004: no silent fallback to ORT)
//	ORACLE_MLX_DYLIB=…  explicit liboraclemlx.dylib path (else ~/.oracle/
//	                    models/lib/ then cpp/oraclemlx/build/)
//
// MLX wins ties: when ORACLE_MLX=1, the ORT path is never touched. Unset it
// to fall back to ORT explicitly.
func MLXEnabled() bool { return os.Getenv("ORACLE_MLX") == "1" }

func oracleModelDir(name string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".oracle", "models", name)
	}
	return filepath.Join(home, ".oracle", "models", name)
}

// --- ORT environment (process-wide, once) ---

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func findORTDylib() (string, error) {
	if p := os.Getenv("ORT_DYLIB"); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("infer: ORT_DYLIB=%s not readable: %w", p, err)
		}
		return p, nil
	}
	var candidates []string
	if p := ortProvisionedDylib(); p != "" {
		candidates = append(candidates, p) // ~/.oracle/lib, from `oracle models pull`
	}
	if runtime.GOOS == "darwin" {
		candidates = append(candidates,
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		)
	} else {
		candidates = append(candidates,
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("infer: onnxruntime shared library not found (tried %v); "+
		"run `oracle models pull` to fetch it, or install one (brew install onnxruntime) / set ORT_DYLIB", candidates)
}

func initORT() error {
	ortInitOnce.Do(func() {
		lib, err := findORTDylib()
		if err != nil {
			ortInitErr = err
			return
		}
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil {
			ortInitErr = fmt.Errorf("infer: initializing onnxruntime (%s): %w", lib, err)
		}
	})
	return ortInitErr
}

// newSessionOptions builds options with CoreML-on-darwin (best effort) and
// 4 intra-op threads on CPU.
func newSessionOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("infer: session options: %w", err)
	}
	if err := opts.SetIntraOpNumThreads(4); err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("infer: SetIntraOpNumThreads: %w", err)
	}
	if runtime.GOOS == "darwin" && os.Getenv("ORACLE_ORT_COREML") == "1" {
		// CoreML EP is opt-in: benchmarked SLOWER than CPU/4-threads on
		// M4 Max for these berts (embed single 200ms vs 120ms), and it cannot
		// compile models with external weight data (judge's model.onnx.data —
		// session creation fails; loadLocalModel then retries CPU-only).
		if err := opts.AppendExecutionProviderCoreML(0); err != nil {
			fmt.Fprintf(os.Stderr, "oracle: CoreML EP unavailable, using CPU: %v\n", err)
		}
	}
	return opts, nil
}

// --- generic local model: session + tokenizer + graph introspection ---

type localModel struct {
	mu          sync.Mutex
	session     *ort.DynamicAdvancedSession
	tok         *wordPieceTokenizer
	inputNames  []string // in graph order
	outputName  string
	outputShape []int64 // graph-declared (may contain dynamic -1)
	seqLen      int     // graph-declared static sequence length, 0 if dynamic
}

func loadLocalModel(dir, wantOutput string) (*localModel, error) {
	if err := initORT(); err != nil {
		return nil, err
	}
	modelPath := filepath.Join(dir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("infer: model missing at %s: %w", modelPath, err)
	}
	tok, err := loadWordPieceTokenizer(filepath.Join(dir, "vocab.txt"))
	if err != nil {
		return nil, err
	}
	// Introspect the graph: input set varies (embedder may omit token_type_ids).
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("infer: introspecting %s: %w", modelPath, err)
	}
	m := &localModel{tok: tok}
	for _, in := range inputs {
		switch in.Name {
		case "input_ids", "attention_mask", "token_type_ids":
			m.inputNames = append(m.inputNames, in.Name)
			// Static sequence axis (e.g. embedder exported at 128) overrides
			// the caller's default max_len.
			if len(in.Dimensions) == 2 && in.Dimensions[1] > 0 {
				m.seqLen = int(in.Dimensions[1])
			}
		default:
			return nil, fmt.Errorf("infer: %s has unexpected input %q", modelPath, in.Name)
		}
	}
	if len(m.inputNames) < 2 {
		return nil, fmt.Errorf("infer: %s has inputs %v; expected at least input_ids+attention_mask",
			modelPath, m.inputNames)
	}
	for _, out := range outputs {
		if out.Name == wantOutput {
			m.outputName = out.Name
			m.outputShape = out.Dimensions
		}
	}
	if m.outputName == "" {
		return nil, fmt.Errorf("infer: %s has no output %q (outputs: %v)", modelPath, wantOutput, outputs)
	}
	newSess := func() (*ort.DynamicAdvancedSession, error) {
		opts, err := newSessionOptions()
		if err != nil {
			return nil, err
		}
		defer opts.Destroy()
		return ort.NewDynamicAdvancedSession(modelPath, m.inputNames, []string{m.outputName}, opts)
	}
	sess, err := newSess()
	if err != nil && runtime.GOOS == "darwin" && os.Getenv("ORACLE_ORT_COREML") == "1" {
		// CoreML EP can't compile external-data models; retry CPU-only.
		fmt.Fprintf(os.Stderr, "oracle: CoreML session failed for %s, retrying CPU-only: %v\n", modelPath, err)
		os.Unsetenv("ORACLE_ORT_COREML")
		sess, err = newSess()
	}
	if err != nil {
		return nil, fmt.Errorf("infer: loading session %s: %w", modelPath, err)
	}
	m.session = sess
	return m, nil
}

// run tokenizes pairs (b[i] may be "") to maxLen and returns the flat float32
// output plus its shape.
func (m *localModel) run(a, b []string, maxLen int) ([]float32, []int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.seqLen > 0 {
		maxLen = m.seqLen
	}
	batch := len(a)
	ids := make([]int64, 0, batch*maxLen)
	mask := make([]int64, 0, batch*maxLen)
	types := make([]int64, 0, batch*maxLen)
	for i := range a {
		pair := ""
		if b != nil {
			pair = b[i]
		}
		ii, am, tt := m.tok.Encode(a[i], pair, maxLen)
		ids = append(ids, ii...)
		mask = append(mask, am...)
		types = append(types, tt...)
	}
	shape := ort.NewShape(int64(batch), int64(maxLen))
	byName := map[string][]int64{"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
	inputs := make([]ort.Value, 0, len(m.inputNames))
	defer func() {
		for _, t := range inputs {
			t.Destroy()
		}
	}()
	for _, name := range m.inputNames {
		t, err := ort.NewTensor(shape, byName[name])
		if err != nil {
			return nil, nil, fmt.Errorf("infer: tensor %s: %w", name, err)
		}
		inputs = append(inputs, t)
	}
	outputs := []ort.Value{nil} // let ORT allocate
	if err := m.session.Run(inputs, outputs); err != nil {
		return nil, nil, fmt.Errorf("infer: session run: %w", err)
	}
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		outputs[0].Destroy()
		return nil, nil, fmt.Errorf("infer: output %s is not float32", m.outputName)
	}
	defer out.Destroy()
	data := out.GetData()
	cp := make([]float32, len(data))
	copy(cp, data)
	return cp, out.GetShape(), nil
}

// --- judge ---

var (
	judgeOnce  sync.Once
	judgeModel *localModel
	judgeErr   error
)

// judgeLocal runs the local 3-class judge on (oldSide, newSide) and returns
// softmax probabilities. Requires ORACLE_LOCAL_JUDGE=1.
func JudgeLocal(oldSide, newSide string) (probs [3]float64, err error) {
	if !localJudgeEnabled() {
		return probs, fmt.Errorf("infer: judgeLocal called but ORACLE_LOCAL_JUDGE!=1")
	}
	if MLXEnabled() {
		return judgeMLX(oldSide, newSide)
	}
	judgeOnce.Do(func() {
		if err := ensureModels("judge_v2_onnx"); err != nil {
			judgeErr = err
			return
		}
		judgeModel, judgeErr = loadLocalModel(oracleModelDir("judge_v2_onnx"), "logits")
	})
	if judgeErr != nil {
		return probs, judgeErr
	}
	out, shape, err := judgeModel.run([]string{oldSide}, []string{newSide}, judgeMaxLen)
	if err != nil {
		return probs, err
	}
	if len(shape) != 2 || shape[1] != 3 || len(out) != 3 {
		return probs, fmt.Errorf("infer: judge output shape %v, want [1 3]", shape)
	}
	return softmax3(out), nil
}

func softmax3(logits []float32) [3]float64 {
	max := float64(logits[0])
	for _, l := range logits[1:] {
		if float64(l) > max {
			max = float64(l)
		}
	}
	var exps [3]float64
	var sum float64
	for i, l := range logits {
		exps[i] = math.Exp(float64(l) - max)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

// --- embedder ---

var (
	embedOnce   sync.Once
	embedModel  *localModel
	embedErrVar error
)

// embedLocal embeds texts with the local e5 model (mean-pool + 512-d
// projection baked into the graph). Requires ORACLE_LOCAL_EMBED=1.
func EmbedLocal(texts []string) ([][]float32, error) {
	if !store.LocalEmbedEnabled() {
		return nil, fmt.Errorf("infer: embedLocal called but ORACLE_LOCAL_EMBED!=1")
	}
	if len(texts) == 0 {
		return nil, fmt.Errorf("infer: embedLocal called with no texts")
	}
	if MLXEnabled() {
		return embedMLX(texts)
	}
	embedOnce.Do(func() {
		if err := ensureModels("embedder_v3_onnx"); err != nil {
			embedErrVar = err
			return
		}
		embedModel, embedErrVar = loadLocalModel(oracleModelDir("embedder_v3_onnx"), "embedding")
	})
	if embedErrVar != nil {
		return nil, embedErrVar
	}
	out, shape, err := embedModel.run(texts, nil, embedMaxLen)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || int(shape[0]) != len(texts) {
		return nil, fmt.Errorf("infer: embed output shape %v, want [%d D]", shape, len(texts))
	}
	dim := int(shape[1])
	if dim != 512 {
		return nil, fmt.Errorf("infer: embed dim %d, want 512", dim)
	}
	res := make([][]float32, len(texts))
	for i := range res {
		res[i] = out[i*dim : (i+1)*dim : (i+1)*dim]
	}
	return res, nil
}
