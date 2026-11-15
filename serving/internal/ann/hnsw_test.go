package ann

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

func randomUnit(rng *rand.Rand, n, d int) [][]float32 {
	out := make([][]float32, n)
	// clustered data (like pin embeddings) is harder for ANN than uniform noise
	centers := make([][]float64, 20)
	for i := range centers {
		centers[i] = make([]float64, d)
		for j := range centers[i] {
			centers[i][j] = rng.NormFloat64()
		}
	}
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		v := make([]float32, d)
		var n2 float64
		for j := range v {
			x := c[j] + 0.6*rng.NormFloat64()
			v[j] = float32(x)
			n2 += x * x
		}
		for j := range v {
			v[j] /= float32(math.Sqrt(n2))
		}
		out[i] = v
	}
	return out
}

func recallAt(t *testing.T, n, d, k, ef int) float64 {
	rng := rand.New(rand.NewSource(1))
	vecs := randomUnit(rng, n, d)
	ids := make([]int, n)
	h := NewHNSW(d, 16, 120, 1)
	for i, v := range vecs {
		ids[i] = i
		h.Insert(i, v)
	}
	queries := randomUnit(rng, 200, d)
	var hit, tot int
	for _, q := range queries {
		truth := map[int]bool{}
		for _, r := range BruteForce(vecs, ids, q, k) {
			truth[r.ID] = true
		}
		for _, r := range h.Search(q, k, ef, nil) {
			if truth[r.ID] {
				hit++
			}
		}
		tot += k
	}
	return float64(hit) / float64(tot)
}

func TestRecallClusteredData(t *testing.T) {
	r := recallAt(t, 5000, 64, 10, 200)
	t.Logf("recall@10 (ef=200) = %.3f", r)
	if r < 0.97 {
		t.Fatalf("recall %.3f below 0.97", r)
	}
}

func TestSearchSkipAndOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	vecs := randomUnit(rng, 500, 16)
	h := NewHNSW(16, 8, 50, 1)
	for i, v := range vecs {
		h.Insert(i, v)
	}
	res := h.Search(vecs[8], 20, 60, func(id int) bool { return id%2 == 1 })
	for i, r := range res {
		if r.ID%2 == 1 {
			t.Fatalf("skipped id %d returned", r.ID)
		}
		if i > 0 && r.Dist < res[i-1].Dist {
			t.Fatal("results not sorted by distance")
		}
	}
	if res[0].ID != 8 {
		t.Fatalf("query vector itself should be nearest, got %d", res[0].ID)
	}
}

func TestEmptyIndex(t *testing.T) {
	if got := NewHNSW(4, 4, 10, 1).Search([]float32{1, 0, 0, 0}, 5, 10, nil); got != nil {
		t.Fatal("empty index should return nil")
	}
}

// Run with -race: concurrent searches while new (fresh) pins are inserted.
func TestConcurrentInsertSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	vecs := randomUnit(rng, 2000, 32)
	h := NewHNSW(32, 8, 60, 1)
	for i := 0; i < 1000; i++ {
		h.Insert(i, vecs[i])
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 1000; i < 2000; i++ {
			h.Insert(i, vecs[i])
		}
	}()
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if len(h.Search(vecs[(g*200+i)%1000], 10, 40, nil)) == 0 {
					t.Error("empty result")
				}
			}
		}(g)
	}
	wg.Wait()
	if h.Len() != 2000 {
		t.Fatalf("len=%d", h.Len())
	}
}

func BenchmarkSearch20k(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	vecs := randomUnit(rng, 20000, 64)
	h := NewHNSW(64, 16, 120, 1)
	for i, v := range vecs {
		h.Insert(i, v)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(vecs[i%len(vecs)], 100, 150, nil)
	}
}

func BenchmarkBruteForce20k(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	vecs := randomUnit(rng, 20000, 64)
	ids := make([]int, len(vecs))
	for i := range ids {
		ids[i] = i
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BruteForce(vecs, ids, vecs[i%len(vecs)], 100)
	}
}
