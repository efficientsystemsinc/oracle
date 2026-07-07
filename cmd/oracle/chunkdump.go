package main

// chunkdump: render chunks from session files WITHOUT ingesting — training
// data for the local extraction model (chunk -> facts distillation).

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"oracle/internal/ingest"
	"os"
)

func chunkDump(args []string) {
	fs := newFlagSet("chunkdump")
	n := fs.Int("n", 4000, "max chunks")
	out := fs.String("out", "chunks.jsonl", "output path")
	_ = fs.Parse(args)
	files := ingest.Discover(0)
	rand.New(rand.NewSource(7)).Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	f, err := os.Create(*out)
	if err != nil {
		fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	written := 0
	for _, sf := range files {
		if written >= *n {
			break
		}
		chunks, err := ingest.ReadNew(sf.Path, sf.Source, 0, "")
		if err != nil {
			continue
		}
		for _, c := range chunks {
			if len(c.Text) < 2000 || written >= *n {
				continue
			}
			_ = enc.Encode(map[string]any{"repo": c.Repo, "event_time": c.EventTime, "text": c.Text})
			written++
		}
	}
	fmt.Println("chunks written:", written)
}
