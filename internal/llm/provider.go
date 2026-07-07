package llm

// Provider config: oracle talks to ANY OpenAI-compatible chat/embeddings
// endpoint. No provider-specific wiring, no silent fallback — config resolves
// env -> ~/.oracle/config -> loud error with setup instructions.
//
// Works verbatim with OpenAI, Azure OpenAI (use the full deployment URL),
// OpenRouter, Together, Ollama (http://localhost:11434/v1/chat/completions,
// key ignored), llama.cpp server, mlx_lm server.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"oracle/internal/store"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type providerConf struct {
	URL   string // full endpoint URL (chat/completions or embeddings)
	Key   string // API key; empty = no auth headers (e.g. Ollama)
	Model string // model name; sent in the request body when set
}

var (
	confOnce sync.Once
	confFile map[string]string
)

// confValue resolves a config key: environment first, then ~/.oracle/config
// (simple KEY=VALUE lines, # comments allowed).
func confValue(name string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	confOnce.Do(func() {
		confFile = map[string]string{}
		b, err := os.ReadFile(filepath.Join(store.OracleHome(), "config"))
		if err != nil {
			return
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			if k, v, ok := strings.Cut(line, "="); ok {
				confFile[strings.TrimSpace(k)] = strings.TrimSpace(v)
			}
		}
	})
	return confFile[name]
}

const llmSetupHelp = `oracle needs an OpenAI-compatible LLM endpoint for extraction/ask.
Set ORACLE_LLM_URL (+ ORACLE_LLM_KEY / ORACLE_LLM_MODEL as needed) in the
environment or in ~/.oracle/config (KEY=VALUE lines). Examples:

  # OpenAI
  ORACLE_LLM_URL=https://api.openai.com/v1/chat/completions
  ORACLE_LLM_KEY=sk-...
  ORACLE_LLM_MODEL=gpt-4.1

  # Ollama (local, no key)
  ORACLE_LLM_URL=http://localhost:11434/v1/chat/completions
  ORACLE_LLM_MODEL=llama3.1

  # Azure OpenAI (full deployment URL; model name is in the URL)
  ORACLE_LLM_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-01-01-preview
  ORACLE_LLM_KEY=<azure key>`

const embedSetupHelp = `oracle needs an embeddings endpoint (or use the LOCAL embedder:
ORACLE_LOCAL_EMBED=1 after 'oracle models pull' — then no remote embed config
is needed at all). For a remote endpoint set ORACLE_EMBED_URL
(+ ORACLE_EMBED_KEY / ORACLE_EMBED_MODEL) in env or ~/.oracle/config:

  # OpenAI
  ORACLE_EMBED_URL=https://api.openai.com/v1/embeddings
  ORACLE_EMBED_KEY=sk-...
  ORACLE_EMBED_MODEL=text-embedding-3-large

  # Azure OpenAI
  ORACLE_EMBED_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/embeddings?api-version=2025-01-01-preview
  ORACLE_EMBED_KEY=<azure key>`

func Config() (providerConf, error) {
	c := providerConf{
		URL:   confValue("ORACLE_LLM_URL"),
		Key:   confValue("ORACLE_LLM_KEY"),
		Model: confValue("ORACLE_LLM_MODEL"),
	}
	if c.URL == "" {
		return c, fmt.Errorf("no LLM endpoint configured.\n\n%s", llmSetupHelp)
	}
	return c, nil
}

func EmbedConfig() (providerConf, error) {
	c := providerConf{
		URL:   confValue("ORACLE_EMBED_URL"),
		Key:   confValue("ORACLE_EMBED_KEY"),
		Model: confValue("ORACLE_EMBED_MODEL"),
	}
	if c.URL == "" {
		return c, fmt.Errorf("no embeddings endpoint configured.\n\n%s", embedSetupHelp)
	}
	return c, nil
}

// isLocalURL reports whether u points at this machine (local model servers
// don't count as frontier-LLM calls).
func isLocalURL(u string) bool {
	p, err := url.Parse(u)
	if err != nil {
		return false
	}
	h := p.Hostname()
	return h == "localhost" || h == "127.0.0.1" || h == "::1"
}

// Body-compat profile, cached per process. Some OpenAI-compatible servers
// 400 on response_format:{type:json_object} or on max_completion_tokens;
// postChat retries once per knob and remembers which profile stuck.
var (
	chatNoRespFormat atomic.Bool // strip response_format
	chatUseMaxTokens atomic.Bool // send max_tokens instead of max_completion_tokens
)

// postChat sends a chat-completions payload to the configured LLM endpoint,
// applying the model name, the reasoning-effort knob, and the cached
// body-compat profile. Retries on 400 by degrading the payload (see above);
// everything else stays loud.
func PostChat(payload map[string]any) ([]byte, error) {
	cfg, err := Config()
	if err != nil {
		return nil, err
	}
	if !isLocalURL(cfg.URL) {
		FrontierCalls.Add(1)
	}
	if cfg.Model != "" {
		payload["model"] = cfg.Model
	}
	for {
		body, err := json.Marshal(applyChatProfile(payload))
		if err != nil {
			return nil, err
		}
		raw, err := PostJSON(cfg.URL, body, cfg.Key)
		if err == nil {
			return raw, nil
		}
		he, ok := err.(*httpError)
		if !ok || he.status != 400 {
			return nil, err
		}
		switch {
		case !chatNoRespFormat.Load() && payload["response_format"] != nil:
			chatNoRespFormat.Store(true)
			fmt.Fprintln(os.Stderr, "oracle: LLM endpoint rejected response_format; retrying without it (cached for this process)")
		case !chatUseMaxTokens.Load() && payload["max_completion_tokens"] != nil:
			chatUseMaxTokens.Store(true)
			fmt.Fprintln(os.Stderr, "oracle: LLM endpoint rejected max_completion_tokens; retrying with max_tokens (cached for this process)")
		default:
			return nil, err
		}
	}
}

// applyChatProfile returns a shallow copy of payload with the cached compat
// profile applied. The original payload is never mutated.
func applyChatProfile(payload map[string]any) map[string]any {
	out := make(map[string]any, len(payload)+1)
	for k, v := range payload {
		out[k] = v
	}
	if chatNoRespFormat.Load() {
		delete(out, "response_format")
	}
	if chatUseMaxTokens.Load() {
		if v, ok := out["max_completion_tokens"]; ok {
			delete(out, "max_completion_tokens")
			out["max_tokens"] = v
		}
	}
	return out
}

// frontierCalls counts chat POSTs to remote (non-localhost) LLM endpoints —
// the $-proxy / zero-frontier check for the askab gate.
var FrontierCalls atomic.Int64

// httpError is a non-200 response; postChat inspects it for body-compat retries.
type httpError struct {
	status int
	body   string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("llm endpoint %d: %s", e.status, store.Truncate(e.body, 300))
}

// postJSON POSTs with up to 3 attempts on transport errors / 429 / 5xx.
// Bounded retry against the SAME endpoint — failures stay loud after that.
// When key is non-empty, both OpenAI-style (Authorization: Bearer) and
// Azure-style (api-key) auth headers are sent; empty key sends none (Ollama).
func PostJSON(url string, body []byte, key string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*attempt) * 5 * time.Second)
		}
		req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
			req.Header.Set("api-key", key)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			lastErr = &httpError{resp.StatusCode, string(raw)}
			continue
		}
		if resp.StatusCode != 200 {
			return nil, &httpError{resp.StatusCode, string(raw)}
		}
		return raw, nil
	}
	return nil, fmt.Errorf("after 3 attempts: %w", lastErr)
}

func ChatJSON(system, user string, maxTokens int, out any) error {
	payload := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"response_format":       map[string]string{"type": "json_object"},
		"max_completion_tokens": maxTokens,
	}
	// only sent when explicitly configured; many servers reject it
	if re := confValue("ORACLE_LLM_REASONING_EFFORT"); re != "" {
		payload["reasoning_effort"] = re
	}
	raw, err := PostChat(payload)
	if err != nil {
		return err
	}
	var cr struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &cr); err != nil {
		return err
	}
	if len(cr.Choices) == 0 || cr.Choices[0].Message.Content == "" {
		return fmt.Errorf("empty completion (finish=%v)", cr.Choices)
	}
	// decode the first JSON object only — models occasionally append junk
	return json.NewDecoder(strings.NewReader(cr.Choices[0].Message.Content)).Decode(out)
}

var httpClient = &http.Client{Timeout: 240 * time.Second}
