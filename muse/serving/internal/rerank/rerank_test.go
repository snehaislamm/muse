package rerank

import (
	"math"
	"testing"
)

func vec(i int) []float32 { v := make([]float32, 4); v[i%4] = 1; return v }

func TestFiltersSpamAndExcluded(t *testing.T) {
	cs := []Candidate{{Pin: 1, Score: 3, Vec: vec(0)}, {Pin: 2, Score: 2, Spam: 0.99, Vec: vec(1)}, {Pin: 3, Score: 1, Vec: vec(2)}}
	out := Apply(cs, 10, Policy{SpamThreshold: 0.9, Lambda: 1}, map[int]bool{3: true})
	if len(out) != 1 || out[0].Pin != 1 {
		t.Fatalf("got %+v", out)
	}
}

func TestMMRPromotesDiversity(t *testing.T) {
	// three near-identical top items and one different item slightly lower
	cs := []Candidate{{Pin: 1, Score: 1.0, Vec: vec(0)}, {Pin: 2, Score: 0.99, Vec: vec(0)},
		{Pin: 3, Score: 0.98, Vec: vec(0)}, {Pin: 4, Score: 0.9, Vec: vec(1)}, {Pin: 5, Score: 0, Vec: vec(2)}}
	pure := Apply(cs, 2, Policy{SpamThreshold: 1, Lambda: 1}, nil)
	div := Apply(cs, 2, Policy{SpamThreshold: 1, Lambda: 0.6}, nil)
	if pure[1].Pin != 2 || div[1].Pin != 4 {
		t.Fatalf("pure=%v div=%v", pure, div)
	}
	if IntraListSimilarity(div) >= IntraListSimilarity(pure) {
		t.Fatal("diversified slate should have lower intra-list similarity")
	}
}

func TestCreatorCap(t *testing.T) {
	var cs []Candidate
	for i := 0; i < 6; i++ {
		cs = append(cs, Candidate{Pin: i, Score: float64(10 - i), Creator: 1 + i/5, Vec: vec(i)})
	}
	out := Apply(cs, 6, Policy{SpamThreshold: 1, Lambda: 1, MaxPerCreator: 2}, nil)
	if len(out) != 3 {
		t.Fatalf("cap 2 for creator 1 plus one item for creator 2 => 3 items, got %d", len(out))
	}
}

func TestToneDiversityBalancesSkewedPool(t *testing.T) {
	// pool mirrors a skewed corpus: 8 pins of range 1, then one each of ranges 2-4
	var cs []Candidate
	for i := 0; i < 8; i++ {
		cs = append(cs, Candidate{Pin: i, Score: 1 - 0.01*float64(i), Tone: 1, Vec: vec(0)})
	}
	for r := 2; r <= 4; r++ {
		cs = append(cs, Candidate{Pin: 10 + r, Score: 0.96, Tone: r, Vec: vec(0)})
	}
	plain := Apply(cs, 6, Policy{SpamThreshold: 1, Lambda: 1}, nil)
	fair := Apply(cs, 6, Policy{SpamThreshold: 1, Lambda: 1, ToneDiversity: 0.6}, nil)
	if ToneEntropy(fair) <= ToneEntropy(plain) {
		t.Fatalf("entropy fair=%.2f plain=%.2f", ToneEntropy(fair), ToneEntropy(plain))
	}
	if !math.IsNaN(ToneEntropy([]Candidate{{Tone: 0}})) {
		t.Fatal("entropy undefined without tone-bearing pins")
	}
}

func TestToneFilterOnlyAffectsToneBearingPins(t *testing.T) {
	cs := []Candidate{{Pin: 1, Score: 1, Tone: 1, Vec: vec(0)}, {Pin: 2, Score: 1, Tone: 3, Vec: vec(1)}, {Pin: 3, Score: 1, Tone: 0, Vec: vec(2)}}
	out := Apply(cs, 5, Policy{SpamThreshold: 1, Lambda: 1, ToneFilter: 3}, nil)
	if len(out) != 2 {
		t.Fatalf("want range-3 pin plus the non-tone pin, got %+v", out)
	}
}
