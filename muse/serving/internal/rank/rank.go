// Package rank computes ranking features and runs the multi-task MLP exported
// by ml/muse_ml/ranker.py. Feature definitions mirror ml/muse_ml/features.py
// exactly; TestParityWithPython guards against training/serving skew.
package rank

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/snehaislam/muse/serving/internal/store"
)

var FeatureNames = []string{"sim_mean", "sim_cluster_max", "visual_sim", "topic_affinity",
	"log_popularity", "age", "spam_score"}

const NumTopics = 16

// UserState is the serving-time user representation, rebuilt from saves.
type UserState struct {
	Empty     bool
	Mean      []float64
	Vis       []float64
	Centroids [][]float64
	Topic     []float64
}

func unit(v []float64) []float64 {
	var n float64
	for _, x := range v {
		n += x * x
	}
	n = math.Sqrt(n)
	if n <= 1e-9 {
		return v
	}
	for i := range v {
		v[i] /= n
	}
	return v
}

// BuildUserState mirrors features.UserState.__init__ (float64 accumulation,
// exponential time decay, cluster centroids from member pins).
func BuildUserState(s *store.Store, saves []store.Save, clusters [][]int, onboardTopics []int) *UserState {
	us := &UserState{Topic: make([]float64, NumTopics)}
	if len(saves) == 0 {
		us.Empty = true
		for _, t := range onboardTopics { // cold start: declared interests only
			if t >= 0 && t < NumTopics {
				us.Topic[t] = 1 / float64(len(onboardTopics))
			}
		}
		return us
	}
	de, dv := s.Emb.Cols, s.Visual.Cols
	us.Mean, us.Vis = make([]float64, de), make([]float64, dv)
	var tsum float64
	for _, sv := range saves {
		w := math.Exp(-math.Ln2 * float64(s.Cfg.RefDay-int(sv.Day)) / s.Cfg.HalfLife)
		for j, x := range s.Emb.Row(int(sv.Pin)) {
			us.Mean[j] += float64(x) * w
		}
		for j, x := range s.Visual.Row(int(sv.Pin)) {
			us.Vis[j] += float64(x) * w
		}
		us.Topic[s.Pin(int(sv.Pin)).TopicID] += w
		tsum += w
	}
	unit(us.Mean)
	unit(us.Vis)
	for i := range us.Topic {
		us.Topic[i] /= tsum
	}
	for _, c := range clusters {
		if len(c) == 0 {
			continue
		}
		cent := make([]float64, de)
		for _, p := range c {
			for j, x := range s.Emb.Row(p) {
				cent[j] += float64(x)
			}
		}
		for j := range cent {
			cent[j] /= float64(len(c))
		}
		us.Centroids = append(us.Centroids, unit(cent))
	}
	if len(us.Centroids) == 0 {
		us.Centroids = [][]float64{us.Mean}
	}
	return us
}

func dot64(a []float64, b []float32) float64 {
	var s float64
	for i := range a {
		s += a[i] * float64(b[i])
	}
	return s
}

// Features fills one row per pin (len(FeatureNames) columns).
func Features(s *store.Store, us *UserState, pins []int) [][]float64 {
	out := make([][]float64, len(pins))
	for i, pid := range pins {
		p := s.Pin(pid)
		f := make([]float64, len(FeatureNames))
		if !us.Empty {
			e := s.Emb.Row(pid)
			f[0] = dot64(us.Mean, e)
			f[1] = math.Inf(-1)
			for _, c := range us.Centroids {
				f[1] = math.Max(f[1], dot64(c, e))
			}
			f[2] = dot64(us.Vis, s.Visual.Row(pid))
		}
		f[3] = us.Topic[p.TopicID] // non-zero for onboarded cold-start users too
		f[4] = math.Log1p(s.Count(pid))
		f[5] = math.Min(math.Max(float64(s.Cfg.RefDay-p.CreatedDay)/90.0, 0), 1)
		f[6] = p.SpamScore
		out[i] = f
	}
	return out
}

// Model is the multi-task MLP: relu(xW1+b1)W2+b2 -> sigmoid per head.
type Model struct {
	Features []string    `json:"features"`
	Heads    []string    `json:"heads"`
	Mu       []float64   `json:"mu"`
	Sd       []float64   `json:"sd"`
	W1       [][]float64 `json:"W1"`
	B1       []float64   `json:"b1"`
	W2       [][]float64 `json:"W2"`
	B2       []float64   `json:"b2"`
}

func LoadModel(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if len(m.Features) != len(FeatureNames) {
		return nil, fmt.Errorf("model expects %d features, server computes %d", len(m.Features), len(FeatureNames))
	}
	for i, f := range m.Features {
		if f != FeatureNames[i] {
			return nil, fmt.Errorf("feature %d: model=%q server=%q", i, f, FeatureNames[i])
		}
	}
	return &m, nil
}

// Predict returns [p_click, p_save, p_hide].
func (m *Model) Predict(x []float64) [3]float64 {
	h := make([]float64, len(m.B1))
	copy(h, m.B1)
	for i, xi := range x {
		xn := (xi - m.Mu[i]) / m.Sd[i]
		for j := range h {
			h[j] += xn * m.W1[i][j]
		}
	}
	var out [3]float64
	for k := 0; k < 3; k++ {
		z := m.B2[k]
		for j, hj := range h {
			if hj > 0 {
				z += hj * m.W2[j][k]
			}
		}
		out[k] = 1 / (1 + math.Exp(-z))
	}
	return out
}

// Utility blends heads into one ranking score. Saves are Pinterest's core
// "long-term value" signal; hides are strongly penalised.
type Weights struct{ Click, Save, Hide float64 }

var DefaultWeights = Weights{Click: 0.3, Save: 1.0, Hide: 2.0}

func (w Weights) Utility(p [3]float64) float64 { return w.Save*p[1] + w.Click*p[0] - w.Hide*p[2] }
