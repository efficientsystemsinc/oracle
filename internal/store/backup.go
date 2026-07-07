package store

// Backups + trace retention. oracle.db is append-only and irreplaceable —
// facts are extracted from session logs that rotate away — so cycle takes a
// daily snapshot. traces is the one unbounded table; keep the newest N.

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

const (
	backupKeep = 7       // daily snapshots retained
	TracesKeep = 100_000 // newest trace rows retained (training substrate — generous)
)

// backupDB snapshots the live db via VACUUM INTO and prunes old snapshots.
// Timestamped names sort lexicographically == chronologically.
func BackupDB(db *sql.DB) (string, error) {
	dir := filepath.Join(OracleHome(), "backups")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dest := filepath.Join(dir, "oracle-"+time.Now().UTC().Format("20060102-150405")+".db")
	if _, err := db.Exec(`VACUUM INTO ?`, dest); err != nil {
		return "", fmt.Errorf("vacuum into %s: %w", dest, err)
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "oracle-*.db"))
	sort.Strings(matches)
	for len(matches) > backupKeep {
		_ = os.Remove(matches[0])
		matches = matches[1:]
	}
	_, _ = db.Exec(`INSERT OR REPLACE INTO meta(k, v) VALUES('last_backup', ?)`,
		strconv.FormatInt(time.Now().Unix(), 10))
	return dest, nil
}

// backupIfDue runs a snapshot at most once per 24h; cycle calls it every pass.
func BackupIfDue(db *sql.DB) (string, error) {
	var v string
	if db.QueryRow(`SELECT v FROM meta WHERE k='last_backup'`).Scan(&v) == nil {
		if ts, err := strconv.ParseInt(v, 10, 64); err == nil && time.Now().Unix()-ts < 86400 {
			return "", nil
		}
	}
	return BackupDB(db)
}

// pruneTraces deletes all but the newest keep rows. Returns rows removed.
func PruneTraces(db *sql.DB, keep int) (int64, error) {
	var cutoff int64
	err := db.QueryRow(`SELECT id FROM traces ORDER BY id DESC LIMIT 1 OFFSET ?`, keep).Scan(&cutoff)
	if err == sql.ErrNoRows {
		return 0, nil // fewer than keep rows
	}
	if err != nil {
		return 0, err
	}
	res, err := db.Exec(`DELETE FROM traces WHERE id <= ?`, cutoff)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}
