package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/snehaislam/muse/serving/internal/feed"
	"github.com/snehaislam/muse/serving/internal/metrics"
	"github.com/snehaislam/muse/serving/internal/rank"
	"github.com/snehaislam/muse/serving/internal/store"
)

var ts *httptest.Server

func TestMain(m *testing.M) {
	dir := "../../../artifacts"
	if _, err := os.Stat(filepath.Join(dir, "pins.json")); err != nil {
		os.Exit(0) // artifacts not built: nothing to integration-test
	}
	s, err := store.Load(dir)
	if err != nil {
		panic(err)
	}
	mod, err := rank.LoadModel(filepath.Join(dir, "ranker.json"))
	if err != nil {
		panic(err)
	}
	ts = httptest.NewServer(New(feed.NewEngine(s, mod, metrics.New()), slog.New(slog.NewTextHandler(io.Discard, nil)), ""))
	code := m.Run()
	ts.Close()
	os.Exit(code)
}

func get(t *testing.T, path string, wantCode int) feed.Response {
	t.Helper()
	r, err := http.Get(ts.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != wantCode {
		b, _ := io.ReadAll(r.Body)
		t.Fatalf("GET %s: status %d (want %d): %s", path, r.StatusCode, wantCode, b)
	}
	var out feed.Response
	_ = json.NewDecoder(r.Body).Decode(&out)
	return out
}

func post(t *testing.T, path, body string) map[string]any {
	t.Helper()
	r, err := http.Post(ts.URL+path, "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(r.Body).Decode(&out)
	if r.StatusCode != 200 {
		t.Fatalf("POST %s: %d %v", path, r.StatusCode, out)
	}
	return out
}

func TestFeedShapeAndInvariants(t *testing.T) {
	for _, v := range []string{"baseline", "ranker", "full"} {
		resp := get(t, "/v1/feed?user=12&n=40&variant="+v, 200)
		if len(resp.Items) != 40 {
			t.Fatalf("%s: got %d items", v, len(resp.Items))
		}
		seen := map[int]bool{}
		for _, it := range resp.Items {
			if seen[it.ID] {
				t.Fatalf("%s: duplicate pin %d", v, it.ID)
			}
			seen[it.ID] = true
		}
	}
	full := get(t, "/v1/feed?user=12&n=40", 200)
	perCreator := map[int]int{}
	for _, it := range full.Items {
		if perCreator[it.Creator]++; perCreator[it.Creator] > 2 {
			t.Fatalf("creator cap violated for creator %d", it.Creator)
		}
	}
}

func TestBadRequests(t *testing.T) {
	get(t, "/v1/feed?user=abc", 400)
	get(t, "/v1/feed?n=10000", 400)
	get(t, "/v1/feed?variant=nope", 400)
	get(t, "/v1/pins/999999/related", 404)
	get(t, "/v1/search", 400)
	r, _ := http.Post(ts.URL+"/v1/events", "application/json", strings.NewReader(`{"user":1,"pin":5,"type":"like"}`))
	if r.StatusCode != 400 {
		t.Fatalf("unknown event type: %d", r.StatusCode)
	}
}

func TestSaveRemovesPinAndHideFilters(t *testing.T) {
	before := get(t, "/v1/feed?user=33&n=10", 200)
	top := before.Items[0].ID
	second := before.Items[1].ID
	post(t, "/v1/events", `{"user":33,"pin":`+itoa(top)+`,"type":"save"}`)
	post(t, "/v1/events", `{"user":33,"pin":`+itoa(second)+`,"type":"hide"}`)
	after := get(t, "/v1/feed?user=33&n=50", 200)
	for _, it := range after.Items {
		if it.ID == top || it.ID == second {
			t.Fatalf("saved/hidden pin %d still in feed", it.ID)
		}
	}
}

func TestColdStartOnboarding(t *testing.T) {
	out := post(t, "/v1/users", `{"topics":[1,2]}`)
	u := int(out["user"].(float64))
	resp := get(t, "/v1/feed?user="+itoa(u)+"&n=20", 200)
	if !resp.Debug.ColdStartUser || len(resp.Items) != 20 {
		t.Fatalf("cold start feed: %+v", resp.Debug)
	}
	for _, it := range resp.Items {
		if it.TopicID != 1 && it.TopicID != 2 {
			t.Fatalf("onboarded topics [1,2], got topic %d", it.TopicID)
		}
	}
	// one save is enough to switch to personalised retrieval
	post(t, "/v1/events", `{"user":`+itoa(u)+`,"pin":`+itoa(resp.Items[0].ID)+`,"type":"save"}`)
	if get(t, "/v1/feed?user="+itoa(u)+"&n=20", 200).Debug.ColdStartUser {
		t.Fatal("user should leave cold start after first save")
	}
}

func TestToneFilterAndSearch(t *testing.T) {
	resp := get(t, "/v1/search?q=braided+hairstyle&n=30&tone=4", 200)
	if len(resp.Items) == 0 {
		t.Fatal("no search results")
	}
	for _, it := range resp.Items {
		if it.Tone != 0 && it.Tone != 4 {
			t.Fatalf("tone filter 4 returned tone %d", it.Tone)
		}
	}
	rel := get(t, "/v1/pins/100/related?n=12", 200)
	if len(rel.Items) != 12 {
		t.Fatalf("related: %d", len(rel.Items))
	}
}

func itoa(i int) string { b, _ := json.Marshal(i); return string(b) }
