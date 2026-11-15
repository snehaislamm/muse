package search

import "testing"

func TestBM25RanksRareTermsHigher(t *testing.T) {
	ix := New()
	docs := []string{"boho living room ideas", "boho bedroom ideas", "sourdough bread for beginners",
		"minimalist living room", "boho reading nook"}
	for i, d := range docs {
		ix.Add(i, d)
	}
	hits := ix.Search("sourdough ideas", 3, nil)
	if hits[0].Doc != 2 {
		t.Fatalf("rare term 'sourdough' should win, got %+v", hits)
	}
	if len(ix.Search("zzz", 3, nil)) != 0 {
		t.Fatal("unknown term should return nothing")
	}
	if got := ix.Search("boho", 5, func(d int) bool { return d == 0 }); len(got) != 2 {
		t.Fatalf("skip not applied: %+v", got)
	}
}

func TestTokenize(t *testing.T) {
	got := Tokenize("Mid-century Living-room, for the 30-minute DINNER!")
	want := []string{"mid-century", "living-room", "30-minute", "dinner"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v", got)
		}
	}
}
