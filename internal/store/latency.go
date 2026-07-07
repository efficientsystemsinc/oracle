package store

// Per-stage latency tracking with p50/p95/p99, exposed on /health.
// Ring buffer per stage; cheap enough to record every request.

import (
	"sort"
	"sync"
	"time"
)

type latRing struct {
	mu   sync.Mutex
	buf  []float64 // ms
	next int
	full bool
}

var latRings sync.Map // stage -> *latRing

func RecordLatency(stage string, start time.Time) {
	ms := float64(time.Since(start).Microseconds()) / 1000.0
	v, _ := latRings.LoadOrStore(stage, &latRing{buf: make([]float64, 2048)})
	r := v.(*latRing)
	r.mu.Lock()
	r.buf[r.next] = ms
	r.next = (r.next + 1) % len(r.buf)
	if r.next == 0 {
		r.full = true
	}
	r.mu.Unlock()
}

func LatencySnapshot() map[string]map[string]float64 {
	out := map[string]map[string]float64{}
	latRings.Range(func(k, v any) bool {
		r := v.(*latRing)
		r.mu.Lock()
		n := r.next
		if r.full {
			n = len(r.buf)
		}
		vals := append([]float64(nil), r.buf[:n]...)
		r.mu.Unlock()
		if len(vals) == 0 {
			return true
		}
		sort.Float64s(vals)
		q := func(p float64) float64 {
			i := int(p * float64(len(vals)-1))
			return vals[i]
		}
		out[k.(string)] = map[string]float64{
			"p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "n": float64(len(vals)),
		}
		return true
	})
	return out
}
