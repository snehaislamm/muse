package pixie

import (
	"math/rand"
	"testing"
)

// two communities joined by one bridge board
func toyGraph() (pinBoards, boardPins [][]int32) {
	boardPins = [][]int32{
		{0, 1, 2, 3}, {1, 2, 3, 4}, {0, 2, 4}, // community A: pins 0-4
		{5, 6, 7, 8}, {6, 7, 8, 9}, {5, 7, 9}, // community B: pins 5-9
		{4, 5}, // bridge
	}
	pinBoards = make([][]int32, 10)
	for b, ps := range boardPins {
		for _, p := range ps {
			pinBoards[p] = append(pinBoards[p], int32(b))
		}
	}
	return
}

func TestWalkStaysInCommunity(t *testing.T) {
	pb, bp := toyGraph()
	res := Recommend(pb, bp, []Query{{Pin: 0, Weight: 1}}, 4, DefaultParams(), rand.New(rand.NewSource(1)), map[int]bool{0: true})
	for _, r := range res {
		if r.Pin > 4 {
			t.Fatalf("top-4 for pin 0 should be in community A, got %v", res)
		}
	}
}

func TestMultiHitBoostsSharedNeighbours(t *testing.T) {
	pb, bp := toyGraph()
	// pins 4 and 5 sit on the bridge: querying one pin from each side should surface them
	qs := []Query{{Pin: 0, Weight: 1}, {Pin: 9, Weight: 1}}
	res := Recommend(pb, bp, qs, 3, DefaultParams(), rand.New(rand.NewSource(2)), map[int]bool{0: true, 9: true})
	if len(res) == 0 {
		t.Fatal("no results")
	}
	seen := map[int]bool{}
	for _, r := range res {
		seen[r.Pin] = true
	}
	if !seen[2] && !seen[7] {
		t.Fatalf("expected central pins of both communities, got %v", res)
	}
}

func TestExcludeAndEmpty(t *testing.T) {
	pb, bp := toyGraph()
	if Recommend(pb, bp, nil, 5, DefaultParams(), rand.New(rand.NewSource(1)), nil) != nil {
		t.Fatal("no queries should give nil")
	}
	ex := map[int]bool{1: true, 2: true, 3: true}
	for _, r := range Recommend(pb, bp, []Query{{Pin: 0, Weight: 1}}, 10, DefaultParams(), rand.New(rand.NewSource(1)), ex) {
		if ex[r.Pin] {
			t.Fatalf("excluded pin %d returned", r.Pin)
		}
	}
}

func TestEarlyStoppingBoundsWork(t *testing.T) {
	pb, bp := toyGraph()
	p := DefaultParams()
	p.NP, p.NV = 2, 2
	sc := &scratch{}
	sc.reset(len(pb))
	touched := walk(pb, bp, 0, 1_000_000, p, rand.New(rand.NewSource(1)), sc)
	total := 0
	for _, pin := range touched {
		total += int(sc.visits[pin])
	}
	if total > 1000 {
		t.Fatalf("early stopping did not trigger: %d visits", total)
	}
}
