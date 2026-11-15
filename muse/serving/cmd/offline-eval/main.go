// Command offline-eval replays held-out saves (days 86-89) against the serving
// stack and reports, per candidate source and per variant:
//
//   - recall of held-out saves (overall and on cold-start pins with no train saves)
//
//   - slate diversity (intra-list similarity), tone-range entropy, catalogue coverage
//
//   - HNSW recall/latency vs exact brute force on the real pin embeddings
//
//     go run ./cmd/offline-eval -artifacts ../artifacts -users 800
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/snehaislam/muse/serving/internal/ann"
	"github.com/snehaislam/muse/serving/internal/feed"
	"github.com/snehaislam/muse/serving/internal/metrics"
	"github.com/snehaislam/muse/serving/internal/rank"
	"github.com/snehaislam/muse/serving/internal/store"
)

func main() {
	dir := flag.String("artifacts", "../artifacts", "artifact dir")
	nUsers := flag.Int("users", 800, "users to evaluate")
	flag.Parse()
	s, err := store.Load(*dir)
	must(err)
	m, err := rank.LoadModel(filepath.Join(*dir, "ranker.json"))
	must(err)
	e := feed.NewEngine(s, m, metrics.New())
	report := map[string]any{}

	// ---- ANN quality ---------------------------------------------------------
	vecs := make([][]float32, s.NumPins())
	ids := make([]int, s.NumPins())
	for i := range vecs {
		vecs[i], ids[i] = s.Emb.Row(i), i
	}
	rng := rand.New(rand.NewSource(5))
	annRows := []map[string]float64{}
	var bruteUs float64
	queries := 300
	for _, ef := range []int{100, 200, 400} {
		var hit, tot int
		var annDur, bruteDur time.Duration
		for qi := 0; qi < queries; qi++ {
			q := vecs[rng.Intn(len(vecs))]
			t0 := time.Now()
			truth := ann.BruteForce(vecs, ids, q, 100)
			bruteDur += time.Since(t0)
			t0 = time.Now()
			got := e.Main.Search(q, 100, ef, nil)
			annDur += time.Since(t0)
			tm := map[int]bool{}
			for _, r := range truth {
				tm[r.ID] = true
			}
			for _, r := range got {
				if tm[r.ID] {
					hit++
				}
			}
			tot += len(truth)
		}
		bruteUs = float64(bruteDur.Microseconds()) / float64(queries)
		annRows = append(annRows, map[string]float64{"ef": float64(ef), "recall_at_100": float64(hit) / float64(tot),
			"hnsw_us_per_query": float64(annDur.Microseconds()) / float64(queries)})
	}
	report["ann"] = map[string]any{"sweep": annRows, "brute_force_us_per_query": bruteUs}

	// ---- held-out saves --------------------------------------------------------
	test := map[int]map[int]bool{}
	for _, sv := range s.TestSaves {
		u, p := int(sv.User), int(sv.Pin)
		if test[u] == nil {
			test[u] = map[int]bool{}
		}
		test[u][p] = true
	}
	users := make([]int, 0, len(test))
	for u := range test {
		users = append(users, u)
	}
	sort.Ints(users)
	rng.Shuffle(len(users), func(i, j int) { users[i], users[j] = users[j], users[i] })
	if len(users) > *nUsers {
		users = users[:*nUsers]
	}
	trainSaved := func(u int) map[int]bool {
		out := map[int]bool{}
		for _, sv := range s.UserSaves(u) {
			out[int(sv.Pin)] = true
		}
		return out
	}

	type agg struct{ hit, tot, coldHit, coldTot float64 }
	sources := map[string]*agg{}
	add := func(name string, got map[int]bool, truth map[int]bool) {
		a := sources[name]
		if a == nil {
			a = &agg{}
			sources[name] = a
		}
		for p := range truth {
			cold := s.Count(p) == 0
			a.tot++
			if cold {
				a.coldTot++
			}
			if got[p] {
				a.hit++
				if cold {
					a.coldHit++
				}
			}
		}
	}
	// per-source candidate recall: run the "full" retrieval once and split by tag
	for _, u := range users {
		seen := trainSaved(u)
		truth := map[int]bool{}
		for p := range test[u] {
			if !seen[p] {
				truth[p] = true
			}
		}
		if len(truth) == 0 {
			continue
		}
		bySrc := map[string]map[int]bool{"union": {}}
		cs := candidatesBySource(e, u)
		for src, pins := range cs {
			bySrc[src] = pins
			for p := range pins {
				bySrc["union"][p] = true
			}
		}
		for name, got := range bySrc {
			add("cand:"+name, got, truth)
		}
		for _, v := range []string{"baseline", "ranker", "full"} {
			r, err := e.Feed(u, 100, v, 0)
			must(err)
			got := map[int]bool{}
			for _, it := range r.Items {
				got[it.ID] = true
			}
			add("slate100:"+v, got, truth)
		}
	}
	rows := map[string]map[string]float64{}
	names := make([]string, 0, len(sources))
	for k, a := range sources {
		names = append(names, k)
		rows[k] = map[string]float64{"recall": a.hit / a.tot, "cold_recall": a.coldHit / math.Max(a.coldTot, 1)}
	}
	sort.Strings(names)
	report["recall"] = rows

	// ---- slate quality -------------------------------------------------------
	slate := map[string]map[string]float64{}
	for _, v := range []string{"baseline", "ranker", "full"} {
		var ils, ent, entN, ms float64
		cover := map[int]bool{}
		for _, u := range users {
			r, _ := e.Feed(u, 50, v, 0)
			ils += r.Debug.IntraListSim
			ms += r.Debug.StageMs["total"]
			if r.Debug.ToneEntropy != nil {
				ent += *r.Debug.ToneEntropy
				entN++
			}
			for _, it := range r.Items {
				cover[it.ID] = true
			}
		}
		n := float64(len(users))
		slate[v] = map[string]float64{"intra_list_similarity": ils / n, "tone_entropy": ent / math.Max(entN, 1),
			"catalog_coverage": float64(len(cover)) / float64(s.NumPins()), "mean_latency_ms": ms / n}
	}
	report["slate"] = slate
	report["users_evaluated"] = len(users)

	out, _ := json.MarshalIndent(report, "", "  ")
	must(os.WriteFile(filepath.Join(*dir, "serving_eval.json"), out, 0o644))
	fmt.Println("| metric | recall | cold-pin recall |\n|---|---|---|")
	for _, k := range names {
		fmt.Printf("| %s | %.3f | %.3f |\n", k, rows[k]["recall"], rows[k]["cold_recall"])
	}
	fmt.Println("\n| variant | ILS (lower=diverse) | tone entropy | coverage | mean ms |\n|---|---|---|---|---|")
	for _, v := range []string{"baseline", "ranker", "full"} {
		r := slate[v]
		fmt.Printf("| %s | %.3f | %.3f | %.3f | %.2f |\n", v, r["intra_list_similarity"], r["tone_entropy"], r["catalog_coverage"], r["mean_latency_ms"])
	}
	fmt.Println("\nANN:", string(mustJSON(report["ann"])))
}

func candidatesBySource(e *feed.Engine, u int) map[string]map[int]bool {
	cs := e.CandidatesFor(u, feed.Variants["full"])
	out := map[string]map[int]bool{}
	for pin, srcs := range cs {
		for _, s := range srcs {
			if out[s] == nil {
				out[s] = map[int]bool{}
			}
			out[s][pin] = true
		}
	}
	return out
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
