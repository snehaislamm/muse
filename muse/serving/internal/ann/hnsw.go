// Package ann implements Hierarchical Navigable Small World graphs
// (Malkov & Yashunin, 2016) for approximate nearest-neighbour search over
// L2-normalised vectors, using cosine distance (1 - dot).
//
// Inserts take a write lock; searches take a read lock, so the index can be
// queried by many request goroutines while fresh pins are added online.
package ann

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
)

type Result struct {
	ID   int
	Dist float32
}

type HNSW struct {
	mu       sync.RWMutex
	dim      int
	m, m0    int
	efConstr int
	ml       float64
	vecs     [][]float32
	ids      []int
	links    [][][]int32 // node -> layer -> neighbours
	entry    int32
	maxLevel int
	rng      *rand.Rand
}

func NewHNSW(dim, m, efConstruction int, seed int64) *HNSW {
	return &HNSW{dim: dim, m: m, m0: 2 * m, efConstr: efConstruction,
		ml: 1 / math.Log(float64(m)), entry: -1, rng: rand.New(rand.NewSource(seed))}
}

func (h *HNSW) Len() int { h.mu.RLock(); defer h.mu.RUnlock(); return len(h.vecs) }

// dist is 1 - dot(a, b), unrolled 4x (the hot loop of every search).
func dist(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return 1 - (s0 + s1 + s2 + s3)
}

// visitedSet is a generation-stamped mark array reused across searches via a
// sync.Pool; it replaces a per-search hash map (map ops were ~20% of CPU).
type visitedSet struct {
	marks []uint32
	gen   uint32
}

var visitedPool = sync.Pool{New: func() any { return &visitedSet{} }}

func (v *visitedSet) reset(n int) {
	if len(v.marks) < n {
		v.marks = make([]uint32, n+n/4)
		v.gen = 0
	}
	v.gen++
	if v.gen == 0 { // wrapped: clear
		clear(v.marks)
		v.gen = 1
	}
}

// visit marks node and reports whether it was already visited.
func (v *visitedSet) visit(node int32) bool {
	if v.marks[node] == v.gen {
		return true
	}
	v.marks[node] = v.gen
	return false
}

// ---- heaps -----------------------------------------------------------------

type item struct {
	node int32
	d    float32
}
type minHeap []item

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].d < h[j].d }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(item)) }
func (h *minHeap) Pop() any          { o := *h; x := o[len(o)-1]; *h = o[:len(o)-1]; return x }

type maxHeap struct{ minHeap }

func (h maxHeap) Less(i, j int) bool { return h.minHeap[i].d > h.minHeap[j].d }

// searchLayer is the greedy beam search of the paper (Algorithm 2).
func (h *HNSW) searchLayer(q []float32, entries []item, ef, layer int) []item {
	visited := visitedPool.Get().(*visitedSet)
	defer visitedPool.Put(visited)
	visited.reset(len(h.vecs))
	cand := &minHeap{}
	res := &maxHeap{}
	for _, e := range entries {
		visited.visit(e.node)
		heap.Push(cand, e)
		heap.Push(res, e)
	}
	for cand.Len() > 0 {
		c := heap.Pop(cand).(item)
		if c.d > res.minHeap[0].d && res.Len() >= ef {
			break
		}
		if layer >= len(h.links[c.node]) {
			continue
		}
		for _, nb := range h.links[c.node][layer] {
			if visited.visit(nb) {
				continue
			}
			d := dist(q, h.vecs[nb])
			if res.Len() < ef || d < res.minHeap[0].d {
				heap.Push(cand, item{nb, d})
				heap.Push(res, item{nb, d})
				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}
	out := append([]item(nil), res.minHeap...)
	sort.Slice(out, func(i, j int) bool { return out[i].d < out[j].d })
	return out
}

// selectNeighbors implements the diversity heuristic (Algorithm 4): keep a
// candidate only if it is closer to the query than to any already-kept
// neighbour. This keeps long-range links and markedly improves recall on
// clustered data like pin embeddings.
func (h *HNSW) selectNeighbors(cands []item, m int) []int32 {
	out := make([]int32, 0, m)
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		ok := true
		for _, o := range out {
			if dist(h.vecs[c.node], h.vecs[o]) < c.d {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, c.node)
		}
	}
	// back-fill with nearest pruned candidates so degree does not collapse
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		dup := false
		for _, o := range out {
			if o == c.node {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, c.node)
		}
	}
	return out
}

// Insert adds a vector with an external id.
func (h *HNSW) Insert(id int, v []float32) {
	h.mu.Lock()
	defer h.mu.Unlock()
	node := int32(len(h.vecs))
	level := int(math.Floor(-math.Log(1-h.rng.Float64()) * h.ml))
	h.vecs = append(h.vecs, v)
	h.ids = append(h.ids, id)
	h.links = append(h.links, make([][]int32, level+1))
	if h.entry < 0 {
		h.entry, h.maxLevel = node, level
		return
	}
	ep := []item{{h.entry, dist(v, h.vecs[h.entry])}}
	for l := h.maxLevel; l > level; l-- {
		ep = h.searchLayer(v, ep, 1, l)[:1]
	}
	for l := min(level, h.maxLevel); l >= 0; l-- {
		cands := h.searchLayer(v, ep, h.efConstr, l)
		maxM := h.m
		if l == 0 {
			maxM = h.m0
		}
		nbs := h.selectNeighbors(cands, h.m)
		h.links[node][l] = nbs
		for _, nb := range nbs {
			h.links[nb][l] = append(h.links[nb][l], node)
			if len(h.links[nb][l]) > maxM { // shrink neighbour list
				cs := make([]item, len(h.links[nb][l]))
				for i, x := range h.links[nb][l] {
					cs[i] = item{x, dist(h.vecs[nb], h.vecs[x])}
				}
				sort.Slice(cs, func(i, j int) bool { return cs[i].d < cs[j].d })
				h.links[nb][l] = h.selectNeighbors(cs, maxM)
			}
		}
		ep = cands
	}
	if level > h.maxLevel {
		h.entry, h.maxLevel = node, level
	}
}

// Search returns the k nearest ids. ef >= k trades latency for recall.
// Ids for which skip returns true are excluded (over-fetching to compensate).
func (h *HNSW) Search(q []float32, k, ef int, skip func(id int) bool) []Result {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry < 0 {
		return nil
	}
	ep := []item{{h.entry, dist(q, h.vecs[h.entry])}}
	for l := h.maxLevel; l > 0; l-- {
		ep = h.searchLayer(q, ep, 1, l)[:1]
	}
	found := h.searchLayer(q, ep, max(ef, k), 0)
	out := make([]Result, 0, k)
	for _, it := range found {
		id := h.ids[it.node]
		if skip != nil && skip(id) {
			continue
		}
		out = append(out, Result{id, it.d})
		if len(out) == k {
			break
		}
	}
	return out
}

// BruteForce is the exact baseline used to measure HNSW recall.
func BruteForce(vecs [][]float32, ids []int, q []float32, k int) []Result {
	all := make([]Result, len(vecs))
	for i, v := range vecs {
		all[i] = Result{ids[i], dist(q, v)}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Dist < all[j].Dist })
	if len(all) > k {
		all = all[:k]
	}
	return all
}
