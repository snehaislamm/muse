// Package rerank turns a ranked candidate list into the final feed slate.
//
// The ranker scores items independently, so its top-N is often 20 near-copies
// of one idea from one creator. The re-ranker applies slate-level policy:
//   - hard filters  (spam above the reviewed threshold, hidden, already saved)
//   - MMR diversity (Carbonell & Goldstein 1998) on the unified embedding
//   - creator cap   (no creator more than MaxPerCreator times in a slate)
//   - inclusive representation: for pins that carry a skin-tone range, a
//     penalty on over-represented ranges so every range gets surfaced, plus an
//     opt-in user filter (like Pinterest's skin tone ranges in search).
package rerank

import (
	"math"
	"sort"
)

type Candidate struct {
	Pin     int
	Score   float64 // ranker utility
	Creator int
	Tone    int // 0 = not applicable, 1..4 = tone range
	Spam    float64
	Vec     []float32
	Sources []string
}

type Policy struct {
	SpamThreshold float64
	Lambda        float64 // MMR relevance weight, 1 = pure relevance
	MaxPerCreator int
	ToneDiversity float64 // 0 disables
	ToneFilter    int     // 0 = off, else only that range for tone-bearing pins
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Apply returns up to n candidates in slate order.
func Apply(cands []Candidate, n int, p Policy, exclude map[int]bool) []Candidate {
	pool := make([]Candidate, 0, len(cands))
	for _, c := range cands {
		if exclude[c.Pin] || c.Spam >= p.SpamThreshold {
			continue
		}
		if p.ToneFilter > 0 && c.Tone > 0 && c.Tone != p.ToneFilter {
			continue
		}
		pool = append(pool, c)
	}
	sort.SliceStable(pool, func(i, j int) bool { return pool[i].Score > pool[j].Score })
	if len(pool) == 0 {
		return nil
	}
	// normalise relevance to [0,1] so lambda is interpretable
	lo, hi := pool[len(pool)-1].Score, pool[0].Score
	rel := func(c Candidate) float64 {
		if hi-lo < 1e-12 {
			return 1
		}
		return (c.Score - lo) / (hi - lo)
	}
	out := make([]Candidate, 0, n)
	used := make([]bool, len(pool))
	perCreator := map[int]int{}
	toneCount := map[int]int{}
	toneTotal := 0
	// maxSim[i] = max cosine similarity of pool[i] to anything already selected,
	// updated incrementally: O(n*k*d) instead of O(n*k^2*d) for the naive MMR loop.
	maxSim := make([]float64, len(pool))
	for len(out) < n {
		best, bestVal := -1, math.Inf(-1)
		for i, c := range pool {
			if used[i] || (p.MaxPerCreator > 0 && perCreator[c.Creator] >= p.MaxPerCreator) {
				continue
			}
			val := rel(c)
			if p.Lambda < 1 {
				val = p.Lambda*val - (1-p.Lambda)*maxSim[i]
			}
			if p.ToneDiversity > 0 && c.Tone > 0 && toneTotal > 0 {
				// share of this range among tone-bearing pins already shown, vs. a uniform 1/4
				val -= p.ToneDiversity * (float64(toneCount[c.Tone])/float64(toneTotal) - 0.25)
			}
			if val > bestVal {
				best, bestVal = i, val
			}
		}
		if best < 0 {
			break
		}
		used[best] = true
		c := pool[best]
		if p.Lambda < 1 {
			for i := range pool {
				if !used[i] {
					maxSim[i] = math.Max(maxSim[i], dot(pool[i].Vec, c.Vec))
				}
			}
		}
		out = append(out, c)
		perCreator[c.Creator]++
		if c.Tone > 0 {
			toneCount[c.Tone]++
			toneTotal++
		}
	}
	return out
}

// ToneEntropy is the normalised Shannon entropy of tone ranges in a slate
// (1 = perfectly balanced across 4 ranges, 0 = a single range).
func ToneEntropy(slate []Candidate) float64 {
	cnt := map[int]float64{}
	var tot float64
	for _, c := range slate {
		if c.Tone > 0 {
			cnt[c.Tone]++
			tot++
		}
	}
	if tot == 0 {
		return math.NaN()
	}
	var h float64
	for _, v := range cnt {
		q := v / tot
		h -= q * math.Log(q)
	}
	return h / math.Log(4)
}

// IntraListSimilarity is the mean pairwise cosine similarity of a slate (lower = more diverse).
func IntraListSimilarity(slate []Candidate) float64 {
	var s float64
	var n int
	for i := range slate {
		for j := i + 1; j < len(slate); j++ {
			s += dot(slate[i].Vec, slate[j].Vec)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return s / float64(n)
}
