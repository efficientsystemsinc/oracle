package store

// Tests for db.go.

import (
	"database/sql"
	"testing"
)

func visibleAt(t *testing.T, db *sql.DB, asOf float64) map[int64]bool {
	t.Helper()
	rows, err := db.Query(`SELECT f.id FROM facts f WHERE `+AsOfPredicate, asOf, asOf)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	out := map[int64]bool{}
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		out[id] = true
	}
	return out
}

func TestAsOfPredicate(t *testing.T) {
	db := TestDB(t)
	// world time: old became true at 100, new (its superseder) at 200.
	// oracle only LEARNED both at 900 — as-of must ignore transaction time.
	old := InsertFact(t, db, "api runs on box-a", "fact", "quasar", 100)
	new_ := InsertFact(t, db, "api runs on box-b", "fact", "quasar", 200)
	Supersede(t, db, old, new_, 900)
	standalone := InsertFact(t, db, "ci uses runner-x", "fact", "quasar", 300)

	cases := []struct {
		asOf float64
		want map[int64]bool
	}{
		{50, map[int64]bool{}},                               // nothing true yet
		{150, map[int64]bool{old: true}},                     // old era
		{200, map[int64]bool{new_: true}},                    // superseder valid exactly at T
		{250, map[int64]bool{new_: true}},                    // new era, old closed by world time
		{1000, map[int64]bool{new_: true, standalone: true}}, // everything current
	}
	for _, c := range cases {
		got := visibleAt(t, db, c.asOf)
		if len(got) != len(c.want) {
			t.Fatalf("as-of %v: got %v want %v", c.asOf, got, c.want)
		}
		for id := range c.want {
			if !got[id] {
				t.Fatalf("as-of %v: missing fact %d (got %v)", c.asOf, id, got)
			}
		}
	}
}
