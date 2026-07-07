package infer

// infer_tokenizer.go — pure-Go WordPiece tokenizer matching HF bert-base-uncased
// (BasicTokenizer with do_lower_case=true + strip accents, then WordPiece with
// "##" continuations). Golden-tested against python transformers in
// infer_tokenizer_test.go.

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	tokCLS = "[CLS]"
	tokSEP = "[SEP]"
	tokPAD = "[PAD]"
	tokUNK = "[UNK]"

	maxWordPieceChars = 100
)

type wordPieceTokenizer struct {
	vocab map[string]int64
	clsID int64
	sepID int64
	padID int64
	unkID int64
}

// loadWordPieceTokenizer reads a BERT vocab.txt (one token per line, id = line index).
func loadWordPieceTokenizer(vocabPath string) (*wordPieceTokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("infer: cannot open vocab %s: %w", vocabPath, err)
	}
	defer f.Close()
	vocab := make(map[string]int64, 32768)
	sc := bufio.NewScanner(f)
	var i int64
	for sc.Scan() {
		vocab[strings.TrimRight(sc.Text(), "\r\n")] = i
		i++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("infer: reading vocab %s: %w", vocabPath, err)
	}
	t := &wordPieceTokenizer{vocab: vocab}
	for _, s := range []struct {
		tok string
		dst *int64
	}{{tokCLS, &t.clsID}, {tokSEP, &t.sepID}, {tokPAD, &t.padID}, {tokUNK, &t.unkID}} {
		id, ok := vocab[s.tok]
		if !ok {
			return nil, fmt.Errorf("infer: vocab %s missing special token %s", vocabPath, s.tok)
		}
		*s.dst = id
	}
	return t, nil
}

// isBertPunct mirrors transformers' _is_punctuation: ASCII symbol ranges are
// treated as punctuation in addition to unicode P*.
func isBertPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// isCJK mirrors transformers' _is_chinese_char ranges.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) || (r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}

// basicTokenize: clean, lowercase, strip accents (NFD, drop Mn), split on
// whitespace/CJK/punctuation.
func basicTokenize(text string) []string {
	var b strings.Builder
	for _, r := range norm.NFD.String(strings.ToLower(text)) {
		switch {
		case r == 0 || r == 0xFFFD || unicode.IsControl(r):
			// drop control chars (tab/newline count as whitespace below in Go's IsSpace? no —
			// IsControl includes \t \n \r, which BERT treats as whitespace)
			if r == '\t' || r == '\n' || r == '\r' {
				b.WriteRune(' ')
			}
		case unicode.Is(unicode.Mn, r):
			// strip accents
		case unicode.IsSpace(r):
			b.WriteRune(' ')
		case isCJK(r):
			b.WriteRune(' ')
			b.WriteRune(r)
			b.WriteRune(' ')
		case isBertPunct(r):
			b.WriteRune(' ')
			b.WriteRune(r)
			b.WriteRune(' ')
		default:
			b.WriteRune(r)
		}
	}
	return strings.Fields(b.String())
}

// wordPiece greedily splits one basic token into vocab pieces; whole token
// becomes [UNK] if any piece fails.
func (t *wordPieceTokenizer) wordPiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > maxWordPieceChars {
		return []int64{t.unkID}
	}
	var out []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var cur int64 = -1
		for end > start {
			piece := string(runes[start:end])
			if start > 0 {
				piece = "##" + piece
			}
			if id, ok := t.vocab[piece]; ok {
				cur = id
				break
			}
			end--
		}
		if cur < 0 {
			return []int64{t.unkID}
		}
		out = append(out, cur)
		start = end
	}
	return out
}

func (t *wordPieceTokenizer) tokenizeToIDs(text string) []int64 {
	var ids []int64
	for _, w := range basicTokenize(text) {
		ids = append(ids, t.wordPiece(w)...)
	}
	return ids
}

// Encode tokenizes text (and optional pair textB, pass "" for none) into
// fixed-length maxLen sequences: [CLS] a [SEP] (b [SEP]) with longest-first
// truncation and [PAD] padding, matching HF truncation="longest_first",
// padding="max_length".
func (t *wordPieceTokenizer) Encode(textA, textB string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	a := t.tokenizeToIDs(textA)
	hasB := textB != ""
	var b []int64
	if hasB {
		b = t.tokenizeToIDs(textB)
	}
	budget := maxLen - 2
	if hasB {
		budget = maxLen - 3
	}
	if budget < 0 {
		budget = 0
	}
	// longest_first: drop one token at a time from the longer sequence.
	for len(a)+len(b) > budget {
		if len(a) >= len(b) {
			a = a[:len(a)-1]
		} else {
			b = b[:len(b)-1]
		}
	}
	inputIDs = make([]int64, 0, maxLen)
	tokenTypeIDs = make([]int64, maxLen)
	attentionMask = make([]int64, maxLen)
	inputIDs = append(inputIDs, t.clsID)
	inputIDs = append(inputIDs, a...)
	inputIDs = append(inputIDs, t.sepID)
	if hasB {
		segStart := len(inputIDs)
		inputIDs = append(inputIDs, b...)
		inputIDs = append(inputIDs, t.sepID)
		for i := segStart; i < len(inputIDs); i++ {
			tokenTypeIDs[i] = 1
		}
	}
	for i := range inputIDs {
		attentionMask[i] = 1
	}
	for len(inputIDs) < maxLen {
		inputIDs = append(inputIDs, t.padID)
	}
	return inputIDs, attentionMask, tokenTypeIDs
}
