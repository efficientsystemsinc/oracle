package llm

// Local extraction backend: ORACLE_LOCAL_EXTRACT=1 routes chatJSON's
// extraction calls to a local mlx_lm server (OpenAI-compatible) instead of
// the remote LLM. Serving: mlx_lm.server --model ~/.oracle/models/extract_mlx --port 8399
// Judge/enrich/ask calls keep their existing routing; only the extraction
// system prompt goes local (it is the volume + cost center).

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func LocalExtractEnabled() bool { return os.Getenv("ORACLE_LOCAL_EXTRACT") == "1" }

func localExtractURL() string {
	if u := os.Getenv("ORACLE_LOCAL_EXTRACT_URL"); u != "" {
		return u
	}
	return "http://127.0.0.1:8399/v1/chat/completions"
}

// chatJSONLocal mirrors chatJSON against the local server. Loud on any error;
// no fallback to the remote LLM (ADR-004) — if the local extractor is on, it is THE path.
func ChatJSONLocal(system, user string, maxTokens int, out any) error {
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  maxTokens,
		"temperature": 0,
	})
	raw, err := PostJSON(localExtractURL(), body, "")
	if err != nil {
		return fmt.Errorf("local extractor: %w", err)
	}
	var cr struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &cr); err != nil {
		return err
	}
	if len(cr.Choices) == 0 || cr.Choices[0].Message.Content == "" {
		return fmt.Errorf("local extractor: empty completion")
	}
	content := cr.Choices[0].Message.Content
	// models sometimes fence JSON; strip fences before decoding
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	return json.NewDecoder(strings.NewReader(content)).Decode(out)
}
