package feed

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/snehaislam/muse/serving/internal/metrics"
	"github.com/snehaislam/muse/serving/internal/rank"
	"github.com/snehaislam/muse/serving/internal/store"
)

func benchEngine(b *testing.B) *Engine {
	dir := "../../../artifacts"
	if _, err := os.Stat(filepath.Join(dir, "pins.json")); err != nil {
		b.Skip("artifacts not built")
	}
	s, err := store.Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	m, err := rank.LoadModel(filepath.Join(dir, "ranker.json"))
	if err != nil {
		b.Fatal(err)
	}
	return NewEngine(s, m, metrics.New())
}

func BenchmarkFeedFull(b *testing.B) {
	e := benchEngine(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Feed(i%4000, 25, "full", 0); err != nil {
			b.Fatal(err)
		}
	}
}
