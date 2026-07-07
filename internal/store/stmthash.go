package store

// Near-dupe fact detection. Parallel subagents in one session restate the same
// finding with cosmetic drift — punctuation, casing, spacing, "0 .58" vs
// "0.58" — and the supersede judge only sometimes catches it. stmt_hash is a
// formatting-invariant fingerprint: two statements that differ only in
// formatting collide; two statements that differ in any letter or DIGIT do
// not (numbers are semantic: recall 0.58 and recall 0.72 are different facts).

import (
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"strings"
	"unicode"
)

// normalizeStatement lowercases and keeps ONLY letters and digits, dropping
// punctuation, whitespace, and number formatting (comma/dot separators). The
// digit values themselves survive, so "recall 0.58" -> "recall058" and
// "recall 0.72" -> "recall072" stay distinct, while "Recall: 0.58!" and
// "recall 0.58" collide.
func normalizeStatement(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func StmtHash(s string) string {
	h := sha1.Sum([]byte(normalizeStatement(s)))
	return hex.EncodeToString(h[:])
}

// backfillStmtHashes fills stmt_hash for rows that predate the column.
// Runs at open; a no-op (one indexed NULL scan) once backfilled.
func backfillStmtHashes(db *sql.DB) error {
	rows, err := db.Query("SELECT id, statement FROM facts WHERE stmt_hash IS NULL")
	if err != nil {
		return err
	}
	type row struct {
		id int64
		h  string
	}
	var todo []row
	for rows.Next() {
		var id int64
		var s string
		if err := rows.Scan(&id, &s); err != nil {
			rows.Close()
			return err
		}
		todo = append(todo, row{id, StmtHash(s)})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	if len(todo) == 0 {
		return nil
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, r := range todo {
		if _, err := tx.Exec("UPDATE facts SET stmt_hash = ? WHERE id = ?", r.h, r.id); err != nil {
			return err
		}
	}
	return tx.Commit()
}
