// Package pixie implements Pixie-style random walks with restart on the
// pin-board bipartite graph (Eksombatchai et al., WWW 2018).
//
//   - Walks start from a weighted set of query pins (a user's recent saves).
//   - Each step goes pin -> random board -> random pin; after every pin the
//     walk restarts with probability alpha (short walks stay "on topic").
//   - Steps are split across query pins in proportion to weight, scaled by a
//     degree term so a viral pin does not dominate the walk budget.
//   - Early stopping: a query's walk ends once nP pins were visited >= nV times.
//   - Multi-hit boosting combines per-query visit counts as (sum sqrt(V_q))^2,
//     rewarding pins reached from *several* of the user's interests.
package pixie

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

type Params struct {
	TotalSteps  int     // walk budget per request
	Alpha       float64 // restart probability after each pin
	NP, NV      int     // early-stopping thresholds
	MaxBoardDeg int     // boards larger than this are subsampled (hub pruning)
}

func DefaultParams() Params {
	return Params{TotalSteps: 30000, Alpha: 0.5, NP: 600, NV: 4, MaxBoardDeg: 500}
}

type Query struct {
	Pin    int
	Weight float64
}

type Scored struct {
	Pin   int
	Score float64
}

// Recommend runs the walks over the given adjacency lists. Pins in exclude
// (already saved / hidden) are never returned.
func Recommend(pinBoards, boardPins [][]int32, qs []Query, k int, p Params, rng *rand.Rand,
	exclude map[int]bool) []Scored {
	if len(qs) == 0 {
		return nil
	}
	// step allocation: s_q = w_q * C(deg_q), C(d) = deg * (Cmax - log deg)
	var cmax float64
	for _, q := range qs {
		cmax = math.Max(cmax, math.Log(float64(len(pinBoards[q.Pin])+1)))
	}
	alloc := make([]float64, len(qs))
	var tot float64
	for i, q := range qs {
		d := float64(len(pinBoards[q.Pin]))
		alloc[i] = q.Weight * d * (cmax + 1 - math.Log(d+1))
		tot += alloc[i]
	}
	sc := scratchPool.Get().(*scratch)
	defer scratchPool.Put(sc)
	sc.reset(len(pinBoards))
	for i, q := range qs {
		if len(pinBoards[q.Pin]) == 0 || tot == 0 {
			continue
		}
		steps := int(float64(p.TotalSteps) * alloc[i] / tot)
		touched := walk(pinBoards, boardPins, int32(q.Pin), steps, p, rng, sc)
		for _, pin := range touched { // multi-hit boosting, then clear per-query counts
			if sc.combined[pin] == 0 {
				sc.hits = append(sc.hits, pin)
			}
			sc.combined[pin] += math.Sqrt(float64(sc.visits[pin]))
			sc.visits[pin] = 0
		}
	}
	out := make([]Scored, 0, len(sc.hits))
	for _, pin := range sc.hits {
		if !exclude[int(pin)] {
			out = append(out, Scored{int(pin), sc.combined[pin] * sc.combined[pin]})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Pin < out[j].Pin
	})
	if len(out) > k {
		out = out[:k]
	}
	return out
}

// scratch holds dense per-pin counters reused across requests (sync.Pool).
// Dense arrays + touched lists replace hash maps, which were ~35% of walk CPU.
type scratch struct {
	visits   []int32
	combined []float64
	touched  []int32
	hits     []int32
}

var scratchPool = sync.Pool{New: func() any { return &scratch{} }}

func (s *scratch) reset(n int) {
	for _, p := range s.hits {
		s.combined[p] = 0
	}
	s.hits = s.hits[:0]
	if len(s.visits) < n {
		s.visits = make([]int32, n+n/4)
		s.combined = make([]float64, n+n/4)
	}
}

func walk(pinBoards, boardPins [][]int32, start int32, steps int, p Params, rng *rand.Rand, sc *scratch) []int32 {
	sc.touched = sc.touched[:0]
	hot := 0
	cur := start
	for s := 0; s < steps; s++ {
		bs := pinBoards[cur]
		if len(bs) == 0 {
			cur = start
			continue
		}
		b := bs[rng.Intn(len(bs))]
		ps := boardPins[b]
		n := len(ps)
		if n > p.MaxBoardDeg {
			n = p.MaxBoardDeg
		}
		next := ps[rng.Intn(n)]
		if next != start {
			if sc.visits[next] == 0 {
				sc.touched = append(sc.touched, next)
			}
			sc.visits[next]++
			if sc.visits[next] == int32(p.NV) {
				hot++
				if hot >= p.NP {
					break
				}
			}
		}
		cur = next
		if rng.Float64() < p.Alpha {
			cur = start
		}
	}
	return sc.touched
}
