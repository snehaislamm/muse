// Package metrics exposes counters and latency histograms in Prometheus text format.
package metrics

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"time"
)

var buckets = []float64{0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1}

type hist struct {
	counts []uint64
	sum    float64
	n      uint64
}

type Registry struct {
	mu       sync.Mutex
	counters map[string]uint64
	hists    map[string]*hist
}

func New() *Registry { return &Registry{counters: map[string]uint64{}, hists: map[string]*hist{}} }

func (r *Registry) Inc(name string, by uint64) {
	r.mu.Lock()
	r.counters[name] += by
	r.mu.Unlock()
}

func (r *Registry) Observe(name string, d time.Duration) {
	s := d.Seconds()
	r.mu.Lock()
	defer r.mu.Unlock()
	h := r.hists[name]
	if h == nil {
		h = &hist{counts: make([]uint64, len(buckets))}
		r.hists[name] = h
	}
	for i, b := range buckets {
		if s <= b {
			h.counts[i]++
		}
	}
	h.sum += s
	h.n++
}

func (r *Registry) Write(w io.Writer) {
	r.mu.Lock()
	defer r.mu.Unlock()
	keys := make([]string, 0, len(r.counters))
	for k := range r.counters {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(w, "muse_%s %d\n", k, r.counters[k])
	}
	hk := make([]string, 0, len(r.hists))
	for k := range r.hists {
		hk = append(hk, k)
	}
	sort.Strings(hk)
	for _, k := range hk {
		h := r.hists[k]
		for i, b := range buckets {
			fmt.Fprintf(w, "muse_%s_seconds_bucket{le=\"%g\"} %d\n", k, b, h.counts[i])
		}
		fmt.Fprintf(w, "muse_%s_seconds_bucket{le=\"+Inf\"} %d\n", k, h.n)
		fmt.Fprintf(w, "muse_%s_seconds_sum %f\nmuse_%s_seconds_count %d\n", k, h.sum, k, h.n)
	}
}
