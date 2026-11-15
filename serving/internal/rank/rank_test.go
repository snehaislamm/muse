package rank

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/snehaislam/muse/serving/internal/store"
)

const artifacts = "../../../artifacts"

// TestParityWithPython recomputes features and predictions for the fixture the
// Python pipeline wrote and requires agreement to 1e-4. A failure here means
// training/serving skew: the model would be served features it never saw.
func TestParityWithPython(t *testing.T) {
	if _, err := os.Stat(filepath.Join(artifacts, "parity_fixture.json")); err != nil {
		t.Skip("artifacts not built; run ml/run_pipeline.py")
	}
	s, err := store.Load(artifacts)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadModel(filepath.Join(artifacts, "ranker.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(artifacts, "parity_fixture.json"))
	var fx []struct {
		User     int         `json:"user"`
		Pins     []int       `json:"pins"`
		Features [][]float64 `json:"features"`
		Pred     [][]float64 `json:"pred"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	maxDiff := 0.0
	for _, c := range fx {
		us := BuildUserState(s, s.UserSaves(c.User), s.UserClusters(c.User), nil)
		got := Features(s, us, c.Pins)
		for i := range c.Pins {
			for j, name := range FeatureNames {
				d := math.Abs(got[i][j] - c.Features[i][j])
				maxDiff = math.Max(maxDiff, d)
				if d > 1e-4 {
					t.Fatalf("user %d pin %d feature %s: go=%.6f py=%.6f", c.User, c.Pins[i], name, got[i][j], c.Features[i][j])
				}
			}
			p := m.Predict(got[i])
			for k := 0; k < 3; k++ {
				if d := math.Abs(p[k] - c.Pred[i][k]); d > 1e-4 {
					t.Fatalf("user %d pin %d head %d: go=%.6f py=%.6f", c.User, c.Pins[i], k, p[k], c.Pred[i][k])
				}
			}
		}
	}
	t.Logf("checked %d users, max feature diff %.2e", len(fx), maxDiff)
}

func TestModelRejectsFeatureMismatch(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "m.json"), []byte(`{"features":["a","b"]}`), 0o644)
	if _, err := LoadModel(filepath.Join(dir, "m.json")); err == nil {
		t.Fatal("expected schema mismatch error")
	}
}

func TestUtilityPenalisesHides(t *testing.T) {
	w := DefaultWeights
	if w.Utility([3]float64{0.5, 0.3, 0.0}) <= w.Utility([3]float64{0.5, 0.3, 0.2}) {
		t.Fatal("higher hide probability must lower utility")
	}
}
