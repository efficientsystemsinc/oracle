package truth

// oracle judgestats: the go/no-go dashboard for flipping the local judge from
// shadow to active. Reads judge_shadow and prints (a) local-vs-the frontier LLM
// agreement overall and by margin bucket (shadow rows only — rows with an
// llm_verdict), (b) active-mode rows taken so far, and (c) the projected
// the frontier LLM pair-call reduction at the current ORACLE_JUDGE_MARGIN threshold
// (fraction of shadow pairs whose local margin clears it).

import (
	"database/sql"
	"fmt"
)

type judgeBucket struct {
	Label    string
	Lo, Hi   float64 // Hi <= 0 means unbounded
	N, Agree int
}

func judgeStatsBuckets() []judgeBucket {
	return []judgeBucket{
		{Label: "  <0.5     ", Lo: -1, Hi: 0.5},
		{Label: "0.5 - 0.7  ", Lo: 0.5, Hi: 0.7},
		{Label: "0.7 - 0.85 ", Lo: 0.7, Hi: 0.85},
		{Label: "0.85 - 0.95", Lo: 0.85, Hi: 0.95},
		{Label: "0.95+      ", Lo: 0.95, Hi: -1},
	}
}

func RunJudgeStats(db *sql.DB) error {
	if err := EnsureJudgeShadow(db); err != nil {
		return err
	}
	threshold, err := LocalJudgeMargin()
	if err != nil {
		return err
	}

	rows, err := db.Query(`SELECT llm_verdict, local_verdict, local_margin FROM judge_shadow`)
	if err != nil {
		return err
	}
	defer rows.Close()

	buckets := judgeStatsBuckets()
	var shadowN, shadowAgree, activeN, aboveThr int
	for rows.Next() {
		var llm, local string
		var margin float64
		if err := rows.Scan(&llm, &local, &margin); err != nil {
			return err
		}
		if llm == "" { // active-mode row: local verdict taken, no LLM comparison
			activeN++
			continue
		}
		shadowN++
		agree := llm == local
		if agree {
			shadowAgree++
		}
		if margin >= threshold {
			aboveThr++
		}
		for i := range buckets {
			if margin >= buckets[i].Lo && (buckets[i].Hi <= 0 || margin < buckets[i].Hi) {
				buckets[i].N++
				if agree {
					buckets[i].Agree++
				}
				break
			}
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}

	fmt.Printf("judge_shadow: %d shadow rows (llm vs local), %d active rows (local-only)\n", shadowN, activeN)
	if shadowN == 0 {
		fmt.Println("no shadow comparisons yet — run with ORACLE_LOCAL_JUDGE=shadow to collect data")
		return nil
	}
	fmt.Printf("overall agreement: %.1f%% (%d/%d)\n\n", 100*float64(shadowAgree)/float64(shadowN), shadowAgree, shadowN)
	fmt.Println("agreement by local margin:")
	for _, b := range buckets {
		if b.N == 0 {
			fmt.Printf("  %s      —      (0 pairs)\n", b.Label)
			continue
		}
		fmt.Printf("  %s  %5.1f%%  (%d/%d)\n", b.Label, 100*float64(b.Agree)/float64(b.N), b.Agree, b.N)
	}
	fmt.Printf("\nat threshold %.2f: %d/%d shadow pairs (%.1f%%) would take the local verdict\n",
		threshold, aboveThr, shadowN, 100*float64(aboveThr)/float64(shadowN))
	fmt.Printf("projected the frontier LLM pair-call reduction in active mode: %.1f%%\n",
		100*float64(aboveThr)/float64(shadowN))
	return nil
}
