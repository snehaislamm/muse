// Package search is a small in-memory BM25 text index over pin titles and topics.
package search

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

type posting struct {
	doc int
	tf  int
}

type Index struct {
	mu     sync.RWMutex
	post   map[string][]posting
	docLen []int
	totLen int
	avgLen float64
	k1, b  float64
}

func Tokenize(s string) []string {
	f := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-'
	})
	out := f[:0]
	for _, t := range f {
		if len(t) > 1 && !stop[t] {
			out = append(out, t)
		}
	}
	return out
}

var stop = map[string]bool{"the": true, "and": true, "for": true, "of": true, "you": true, "ll": true, "to": true, "a": true}

func New() *Index { return &Index{post: map[string][]posting{}, k1: 1.2, b: 0.75} }

// Add indexes a document. Docs must be added with increasing ids.
func (ix *Index) Add(doc int, text string) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	toks := Tokenize(text)
	for len(ix.docLen) <= doc {
		ix.docLen = append(ix.docLen, 0)
	}
	ix.totLen += len(toks) - ix.docLen[doc]
	ix.docLen[doc] = len(toks)
	tf := map[string]int{}
	for _, t := range toks {
		tf[t]++
	}
	for t, c := range tf {
		ix.post[t] = append(ix.post[t], posting{doc, c})
	}
	ix.avgLen = float64(ix.totLen) / float64(len(ix.docLen))
}

type Hit struct {
	Doc   int
	Score float64
}

func (ix *Index) Search(q string, k int, skip func(int) bool) []Hit {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	scores := map[int]float64{}
	N := float64(len(ix.docLen))
	for _, t := range Tokenize(q) {
		ps := ix.post[t]
		if len(ps) == 0 {
			continue
		}
		idf := math.Log(1 + (N-float64(len(ps))+0.5)/(float64(len(ps))+0.5))
		for _, p := range ps {
			tf := float64(p.tf)
			norm := tf * (ix.k1 + 1) / (tf + ix.k1*(1-ix.b+ix.b*float64(ix.docLen[p.doc])/ix.avgLen))
			scores[p.doc] += idf * norm
		}
	}
	out := make([]Hit, 0, len(scores))
	for d, s := range scores {
		if skip == nil || !skip(d) {
			out = append(out, Hit{d, s})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Doc < out[j].Doc
	})
	if len(out) > k {
		out = out[:k]
	}
	return out
}
