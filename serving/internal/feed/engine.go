// Package feed orchestrates the recommendation funnel:
//
//	candidate generation (ANN mean, ANN per-interest, Pixie walk, fresh, popular)
//	  -> feature computation + multi-task MLP ranking
//	  -> slate re-ranking (filters, diversity, creator cap, inclusive tone)
//
// Variants make every stage switchable so the A/B simulator can attribute
// online wins to individual components.
package feed

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/snehaislam/muse/serving/internal/ann"
	"github.com/snehaislam/muse/serving/internal/metrics"
	"github.com/snehaislam/muse/serving/internal/pixie"
	"github.com/snehaislam/muse/serving/internal/rank"
	"github.com/snehaislam/muse/serving/internal/rerank"
	"github.com/snehaislam/muse/serving/internal/search"
	"github.com/snehaislam/muse/serving/internal/store"
)

const FreshWindowDays = 7

type Variant struct {
	Name        string
	UseClusters bool
	UsePixie    bool
	UseFresh    bool
	UseRanker   bool // false: rank by embedding similarity only
	Lambda      float64
	CreatorCap  int
	ToneDiv     float64
	Model       string // "" = default (IPW-debiased) ranker; "naive" = trained on raw logged labels
}

var Variants = map[string]Variant{
	"baseline": {Name: "baseline", Lambda: 1},
	"ranker":   {Name: "ranker", UseClusters: true, UsePixie: true, UseRanker: true, Lambda: 1},
	"full": {Name: "full", UseClusters: true, UsePixie: true, UseFresh: true, UseRanker: true,
		Lambda: 0.85, CreatorCap: 2, ToneDiv: 0.6},
	// identical to "full" but ranked by the model trained without position-bias correction
	"full_naive": {Name: "full_naive", UseClusters: true, UsePixie: true, UseFresh: true, UseRanker: true,
		Lambda: 0.85, CreatorCap: 2, ToneDiv: 0.6, Model: "naive"},
}

const DefaultVariant = "full"

type Engine struct {
	S       *store.Store
	Model   *rank.Model            // default ranker
	Alt     map[string]*rank.Model // alternative rankers addressable by Variant.Model
	Main    *ann.HNSW
	Fresh   *ann.HNSW
	Text    *search.Index
	M       *metrics.Registry
	Pixie   pixie.Params
	Weights rank.Weights

	mu      sync.RWMutex
	onboard map[int][]int
	seq     atomic.Int64
}

func f32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// NewEngine builds the ANN and text indexes (a few seconds for 20k pins).
func NewEngine(s *store.Store, m *rank.Model, reg *metrics.Registry) *Engine {
	e := &Engine{S: s, Model: m, M: reg, Pixie: pixie.DefaultParams(), Weights: rank.DefaultWeights,
		Main: ann.NewHNSW(s.Emb.Cols, 16, 120, 1), Fresh: ann.NewHNSW(s.Emb.Cols, 12, 80, 2),
		Text: search.New(), onboard: map[int][]int{}, Alt: map[string]*rank.Model{}}
	for id := 0; id < s.NumPins(); id++ {
		p := s.Pin(id)
		e.Main.Insert(id, s.Emb.Row(id))
		if p.CreatedDay >= s.Cfg.RefDay-FreshWindowDays {
			e.Fresh.Insert(id, s.Emb.Row(id))
		}
		e.Text.Add(id, p.Title+" "+p.Topic)
	}
	return e
}

func (e *Engine) SetOnboarding(u int, topics []int) {
	e.mu.Lock()
	e.onboard[u] = topics
	e.mu.Unlock()
}

func (e *Engine) Onboarding(u int) []int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.onboard[u]
}

type Item struct {
	store.Pin
	Score   float64  `json:"score"`
	PSave   float64  `json:"p_save"`
	PClick  float64  `json:"p_click"`
	PHide   float64  `json:"p_hide"`
	Sources []string `json:"sources"`
}

type Debug struct {
	Variant       string             `json:"variant"`
	Candidates    map[string]int     `json:"candidates"`
	TotalCands    int                `json:"total_candidates"`
	StageMs       map[string]float64 `json:"stage_ms"`
	ToneEntropy   *float64           `json:"tone_entropy,omitempty"`
	IntraListSim  float64            `json:"intra_list_similarity"`
	ColdStartUser bool               `json:"cold_start_user"`
}

type Response struct {
	Items []Item `json:"items"`
	Debug Debug  `json:"debug"`
}

type userCtx struct {
	saves   []store.Save
	state   *rank.UserState
	exclude map[int]bool
}

func (e *Engine) user(u int) *userCtx {
	saves := e.S.UserSaves(u)
	ex := e.S.Hidden(u)
	for _, sv := range saves {
		ex[int(sv.Pin)] = true
	}
	return &userCtx{saves, rank.BuildUserState(e.S, saves, e.S.UserClusters(u), e.Onboarding(u)), ex}
}

type candSet struct {
	order []int
	src   map[int][]string
	count map[string]int
}

func newCandSet() *candSet { return &candSet{src: map[int][]string{}, count: map[string]int{}} }

func (c *candSet) add(pin int, source string) {
	if _, ok := c.src[pin]; !ok {
		c.order = append(c.order, pin)
	}
	c.src[pin] = append(c.src[pin], source)
	c.count[source]++
}

// Candidates runs every enabled retrieval source and returns the union.
func (e *Engine) Candidates(u int, uc *userCtx, v Variant) *candSet {
	cs := newCandSet()
	skip := func(id int) bool { return uc.exclude[id] }
	st := uc.state
	if st.Empty {
		topics := e.Onboarding(u)
		if len(topics) == 0 {
			for t := 0; t < e.S.NumTopics(); t++ {
				topics = append(topics, t)
			}
		}
		per := max(300/len(topics), 10)
		for _, t := range topics {
			for _, p := range e.S.TopPinsInTopic(t, per) {
				cs.add(p, "popular")
			}
		}
		return cs
	}
	q := f32(st.Mean)
	for _, r := range e.Main.Search(q, 200, 260, skip) {
		cs.add(r.ID, "ann_mean")
	}
	if v.UseClusters {
		for i, c := range st.Centroids {
			if i >= 3 {
				break
			}
			for _, r := range e.Main.Search(f32(c), 60, 100, skip) {
				cs.add(r.ID, "ann_interest")
			}
		}
	}
	if v.UsePixie {
		qs := pixieQueries(uc.saves, e.S.Cfg)
		rng := rand.New(rand.NewSource(int64(u)*7919 + e.seq.Add(1)))
		e.S.WithGraph(func(pb, bp [][]int32) {
			for _, r := range pixie.Recommend(pb, bp, qs, 150, e.Pixie, rng, uc.exclude) {
				cs.add(r.Pin, "pixie")
			}
		})
	}
	if v.UseFresh {
		for _, r := range e.Fresh.Search(q, 60, 100, skip) {
			cs.add(r.ID, "fresh")
		}
	}
	return cs
}

// CandidatesFor exposes retrieval (pin -> source tags) for offline evaluation.
func (e *Engine) CandidatesFor(u int, v Variant) map[int][]string {
	return e.Candidates(u, e.user(u), v).src
}

// pixieQueries: the 20 most recent distinct saves, weighted by time decay.
func pixieQueries(saves []store.Save, cfg store.Config) []pixie.Query {
	sorted := append([]store.Save(nil), saves...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Day > sorted[j].Day })
	seen := map[int32]bool{}
	var qs []pixie.Query
	for _, sv := range sorted {
		if seen[sv.Pin] {
			continue
		}
		seen[sv.Pin] = true
		w := math.Exp(-math.Ln2 * float64(cfg.RefDay-int(sv.Day)) / cfg.HalfLife)
		qs = append(qs, pixie.Query{Pin: int(sv.Pin), Weight: w})
		if len(qs) == 20 {
			break
		}
	}
	return qs
}

func (e *Engine) score(uc *userCtx, pins []int, v Variant) ([]float64, [][3]float64) {
	feats := rank.Features(e.S, uc.state, pins)
	scores := make([]float64, len(pins))
	probs := make([][3]float64, len(pins))
	model := e.Model
	if m, ok := e.Alt[v.Model]; ok && v.Model != "" {
		model = m
	}
	for i, f := range feats {
		probs[i] = model.Predict(f)
		if v.UseRanker {
			scores[i] = e.Weights.Utility(probs[i])
		} else if uc.state.Empty {
			scores[i] = f[4] // popularity for cold users
		} else {
			scores[i] = f[0] // embedding similarity baseline
		}
	}
	return scores, probs
}

func (e *Engine) toItems(slate []rerank.Candidate, probs map[int][3]float64) []Item {
	out := make([]Item, len(slate))
	for i, c := range slate {
		p := probs[c.Pin]
		pin := e.S.Pin(c.Pin)
		pin.SaveCount = int(e.S.Count(c.Pin))
		out[i] = Item{Pin: pin, Score: c.Score, PClick: p[0], PSave: p[1], PHide: p[2], Sources: c.Sources}
	}
	return out
}

func (e *Engine) slate(cs *candSet, scores []float64, probs [][3]float64, n int, v Variant, tone int,
	exclude map[int]bool) ([]rerank.Candidate, map[int][3]float64) {
	cands := make([]rerank.Candidate, len(cs.order))
	pm := make(map[int][3]float64, len(cs.order))
	for i, pid := range cs.order {
		p := e.S.Pin(pid)
		cands[i] = rerank.Candidate{Pin: pid, Score: scores[i], Creator: p.Creator, Tone: p.Tone,
			Spam: p.SpamScore, Vec: e.S.Emb.Row(pid), Sources: cs.src[pid]}
		pm[pid] = probs[i]
	}
	pol := rerank.Policy{SpamThreshold: e.S.Cfg.SpamThreshold, Lambda: v.Lambda, MaxPerCreator: v.CreatorCap,
		ToneDiversity: v.ToneDiv, ToneFilter: tone}
	return rerank.Apply(cands, n, pol, exclude), pm
}

// Feed returns the home feed for a user.
func (e *Engine) Feed(u, n int, variantName string, tone int) (*Response, error) {
	v, ok := Variants[variantName]
	if !ok {
		return nil, fmt.Errorf("unknown variant %q", variantName)
	}
	return e.FeedWith(u, n, v, tone)
}

// FeedWith runs the funnel with an explicit (possibly overridden) variant config.
func (e *Engine) FeedWith(u, n int, v Variant, tone int) (*Response, error) {
	t0 := time.Now()
	stage := map[string]float64{}
	mark := func(name string, t time.Time) time.Time {
		now := time.Now()
		stage[name] = float64(now.Sub(t).Microseconds()) / 1000
		return now
	}
	uc := e.user(u)
	t := mark("user_state", t0)
	cs := e.Candidates(u, uc, v)
	t = mark("retrieval", t)
	scores, probs := e.score(uc, cs.order, v)
	t = mark("ranking", t)
	slate, pm := e.slate(cs, scores, probs, n, v, tone, uc.exclude)
	mark("rerank", t)
	stage["total"] = float64(time.Since(t0).Microseconds()) / 1000
	for src, c := range cs.count {
		e.M.Inc("candidates_"+src+"_total", uint64(c))
	}
	d := Debug{Variant: v.Name, Candidates: cs.count, TotalCands: len(cs.order), StageMs: stage,
		IntraListSim: rerank.IntraListSimilarity(slate), ColdStartUser: uc.state.Empty}
	if h := rerank.ToneEntropy(slate); !math.IsNaN(h) {
		d.ToneEntropy = &h
	}
	return &Response{Items: e.toItems(slate, pm), Debug: d}, nil
}

// Related powers the pin close-up "More ideas like this": visual/unified ANN
// neighbours plus a single-pin Pixie walk, blended by reciprocal rank.
func (e *Engine) Related(pin, n, u int) (*Response, error) {
	if pin < 0 || pin >= e.S.NumPins() {
		return nil, fmt.Errorf("unknown pin %d", pin)
	}
	t0 := time.Now()
	uc := e.user(u)
	uc.exclude[pin] = true
	cs := newCandSet()
	rr := map[int]float64{}
	for i, r := range e.Main.Search(e.S.Emb.Row(pin), 150, 200, func(id int) bool { return id == pin }) {
		cs.add(r.ID, "ann_related")
		rr[r.ID] += 1 / float64(60+i)
	}
	rng := rand.New(rand.NewSource(int64(pin) + e.seq.Add(1)))
	e.S.WithGraph(func(pb, bp [][]int32) {
		for i, r := range pixie.Recommend(pb, bp, []pixie.Query{{Pin: pin, Weight: 1}}, 150, e.Pixie, rng,
			map[int]bool{pin: true}) {
			cs.add(r.Pin, "pixie")
			rr[r.Pin] += 1 / float64(60+i)
		}
	})
	_, probs := e.score(uc, cs.order, Variants["full"])
	scores := make([]float64, len(cs.order))
	for i, p := range cs.order {
		scores[i] = rr[p]
	}
	slate, pm := e.slate(cs, scores, probs, n, Variant{Lambda: 0.9, CreatorCap: 3, ToneDiv: 0.6}, 0, uc.exclude)
	return &Response{Items: e.toItems(slate, pm), Debug: Debug{Variant: "related", Candidates: cs.count,
		TotalCands: len(cs.order), StageMs: map[string]float64{"total": float64(time.Since(t0).Microseconds()) / 1000},
		IntraListSim: rerank.IntraListSimilarity(slate)}}, nil
}

// Search: BM25 text relevance, personalised by the ranker when a user is known.
func (e *Engine) Search(query string, n, u, tone int) (*Response, error) {
	t0 := time.Now()
	uc := e.user(u)
	hits := e.Text.Search(query, 300, nil)
	cs := newCandSet()
	bm := make([]float64, len(hits))
	for i, h := range hits {
		cs.add(h.Doc, "bm25")
		bm[i] = h.Score
	}
	if len(hits) == 0 {
		return &Response{Items: []Item{}, Debug: Debug{Variant: "search", StageMs: map[string]float64{}}}, nil
	}
	util, probs := e.score(uc, cs.order, Variants["full"])
	scores := make([]float64, len(hits))
	top := bm[0]
	for i := range hits {
		scores[i] = 0.6*bm[i]/top + 0.4*util[i]
	}
	slate, pm := e.slate(cs, scores, probs, n, Variant{Lambda: 0.9, CreatorCap: 2, ToneDiv: 0.6}, tone, nil)
	d := Debug{Variant: "search", Candidates: cs.count, TotalCands: len(hits),
		StageMs:      map[string]float64{"total": float64(time.Since(t0).Microseconds()) / 1000},
		IntraListSim: rerank.IntraListSimilarity(slate)}
	if h := rerank.ToneEntropy(slate); !math.IsNaN(h) {
		d.ToneEntropy = &h
	}
	return &Response{Items: e.toItems(slate, pm), Debug: d}, nil
}
