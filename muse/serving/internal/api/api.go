// Package api exposes the engine over HTTP/JSON.
//
//	GET  /v1/feed?user=&n=&variant=&tone=     home feed
//	GET  /v1/pins/{id}                         pin
//	GET  /v1/pins/{id}/related?user=&n=        "More ideas like this"
//	GET  /v1/search?q=&user=&n=&tone=          text search
//	POST /v1/events   {user,pin,type,board}    save | hide | click
//	POST /v1/users    {topics:[...]}           sign-up with onboarding topics
//	GET  /v1/users/{id}/boards                 boards with pins
//	GET  /v1/topics                            topic list for onboarding
//	GET  /healthz, /metrics
package api

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strconv"
	"time"

	"github.com/snehaislam/muse/serving/internal/feed"
	"github.com/snehaislam/muse/serving/internal/store"
)

type Server struct {
	E   *feed.Engine
	Log *slog.Logger
	mux *http.ServeMux
}

func New(e *feed.Engine, log *slog.Logger, staticDir string) *Server {
	s := &Server{E: e, Log: log, mux: http.NewServeMux()}
	s.handle("GET /v1/feed", "feed", s.feed)
	s.handle("GET /v1/pins/{id}", "pin", s.pin)
	s.handle("GET /v1/pins/{id}/related", "related", s.related)
	s.handle("GET /v1/search", "search", s.search)
	s.handle("POST /v1/events", "events", s.events)
	s.handle("POST /v1/users", "signup", s.signup)
	s.handle("GET /v1/users/{id}/boards", "boards", s.boards)
	s.handle("GET /v1/topics", "topics", s.topics)
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	s.mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		e.M.Write(w)
	})
	if staticDir != "" {
		s.mux.Handle("GET /", http.FileServer(http.Dir(staticDir)))
	}
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	s.mux.ServeHTTP(w, r)
}

type httpError struct {
	code int
	msg  string
}

func (e httpError) Error() string { return e.msg }

func badRequest(msg string) error { return httpError{http.StatusBadRequest, msg} }

// handle wraps a handler with panic recovery, latency metrics and structured logs.
func (s *Server) handle(pattern, name string, h func(*http.Request) (any, error)) {
	s.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		status := http.StatusOK
		defer func() {
			if rec := recover(); rec != nil {
				s.Log.Error("panic", "endpoint", name, "err", rec, "stack", string(debug.Stack()))
				status = http.StatusInternalServerError
				writeJSON(w, status, map[string]string{"error": "internal error"})
			}
			s.E.M.Observe("http_"+name, time.Since(t0))
			s.E.M.Inc("http_"+name+"_status_"+strconv.Itoa(status), 1)
			s.Log.Debug("request", "endpoint", name, "status", status, "ms", time.Since(t0).Milliseconds())
		}()
		out, err := h(r)
		if err != nil {
			var he httpError
			status = http.StatusInternalServerError
			if errors.As(err, &he) {
				status = he.code
			}
			writeJSON(w, status, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, status, out)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func intParam(r *http.Request, key string, def, lo, hi int) (int, error) {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < lo || n > hi {
		return 0, badRequest(key + " must be an integer in [" + strconv.Itoa(lo) + "," + strconv.Itoa(hi) + "]")
	}
	return n, nil
}

func (s *Server) pinID(r *http.Request) (int, error) {
	id, err := strconv.Atoi(r.PathValue("id"))
	if err != nil || id < 0 || id >= s.E.S.NumPins() {
		return 0, httpError{http.StatusNotFound, "pin not found"}
	}
	return id, nil
}

func (s *Server) feed(r *http.Request) (any, error) {
	u, err := intParam(r, "user", 0, 0, 1<<30)
	if err != nil {
		return nil, err
	}
	n, err := intParam(r, "n", 50, 1, 200)
	if err != nil {
		return nil, err
	}
	tone, err := intParam(r, "tone", 0, 0, 4)
	if err != nil {
		return nil, err
	}
	v := r.URL.Query().Get("variant")
	if v == "" {
		v = feed.DefaultVariant
	}
	if _, ok := feed.Variants[v]; !ok {
		return nil, badRequest("unknown variant")
	}
	vv := feed.Variants[v]
	// experiment overrides (used by the A/B simulator to sweep re-ranking knobs)
	q := r.URL.Query()
	if x := q.Get("lambda"); x != "" {
		f, err := strconv.ParseFloat(x, 64)
		if err != nil || f < 0 || f > 1 {
			return nil, badRequest("lambda must be in [0,1]")
		}
		vv.Lambda = f
	}
	if x := q.Get("cap"); x != "" {
		c, err := strconv.Atoi(x)
		if err != nil || c < 0 || c > 50 {
			return nil, badRequest("cap must be in [0,50]")
		}
		vv.CreatorCap = c
	}
	if x := q.Get("tonediv"); x != "" {
		f, err := strconv.ParseFloat(x, 64)
		if err != nil || f < 0 || f > 5 {
			return nil, badRequest("tonediv must be in [0,5]")
		}
		vv.ToneDiv = f
	}
	return s.E.FeedWith(u, n, vv, tone)
}

func (s *Server) pin(r *http.Request) (any, error) {
	id, err := s.pinID(r)
	if err != nil {
		return nil, err
	}
	p := s.E.S.Pin(id)
	p.SaveCount = int(s.E.S.Count(id))
	return p, nil
}

func (s *Server) related(r *http.Request) (any, error) {
	id, err := s.pinID(r)
	if err != nil {
		return nil, err
	}
	u, err := intParam(r, "user", 0, 0, 1<<30)
	if err != nil {
		return nil, err
	}
	n, err := intParam(r, "n", 30, 1, 200)
	if err != nil {
		return nil, err
	}
	return s.E.Related(id, n, u)
}

func (s *Server) search(r *http.Request) (any, error) {
	q := r.URL.Query().Get("q")
	if q == "" || len(q) > 200 {
		return nil, badRequest("q is required (max 200 chars)")
	}
	u, err := intParam(r, "user", 0, 0, 1<<30)
	if err != nil {
		return nil, err
	}
	n, err := intParam(r, "n", 40, 1, 200)
	if err != nil {
		return nil, err
	}
	tone, err := intParam(r, "tone", 0, 0, 4)
	if err != nil {
		return nil, err
	}
	return s.E.Search(q, n, u, tone)
}

type event struct {
	User  int    `json:"user"`
	Pin   int    `json:"pin"`
	Type  string `json:"type"`
	Board int    `json:"board"`
}

func (s *Server) events(r *http.Request) (any, error) {
	var ev event
	ev.Board = -1
	if err := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<14)).Decode(&ev); err != nil {
		return nil, badRequest("invalid JSON body")
	}
	if ev.Pin < 0 || ev.Pin >= s.E.S.NumPins() || ev.User < 0 {
		return nil, badRequest("unknown user or pin")
	}
	s.E.M.Inc("events_"+ev.Type+"_total", 1)
	switch ev.Type {
	case "save":
		b, err := s.E.S.RecordSave(ev.User, ev.Pin, ev.Board)
		if err != nil {
			return nil, badRequest(err.Error())
		}
		return map[string]any{"ok": true, "board": b}, nil
	case "hide":
		s.E.S.RecordHide(ev.User, ev.Pin)
	case "click":
	default:
		return nil, badRequest("type must be save, hide or click")
	}
	return map[string]any{"ok": true}, nil
}

func (s *Server) signup(r *http.Request) (any, error) {
	var body struct {
		Topics []int `json:"topics"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<12)).Decode(&body); err != nil {
		return nil, badRequest("invalid JSON body")
	}
	for _, t := range body.Topics {
		if t < 0 || t >= s.E.S.NumTopics() {
			return nil, badRequest("unknown topic id")
		}
	}
	u := s.E.S.NewUser()
	s.E.SetOnboarding(u, body.Topics)
	return map[string]int{"user": u}, nil
}

func (s *Server) boards(r *http.Request) (any, error) {
	u, err := strconv.Atoi(r.PathValue("id"))
	if err != nil || u < 0 {
		return nil, badRequest("bad user id")
	}
	type boardOut struct {
		store.Board
		Pins []store.Pin `json:"pins"`
	}
	out := []boardOut{}
	for _, b := range s.E.S.UserBoards(u) {
		bo := boardOut{Board: b, Pins: []store.Pin{}}
		for _, p := range s.E.S.BoardPins(b.ID) {
			bo.Pins = append(bo.Pins, s.E.S.Pin(int(p)))
		}
		out = append(out, bo)
	}
	return out, nil
}

func (s *Server) topics(*http.Request) (any, error) {
	names := make([]map[string]any, s.E.S.NumTopics())
	for t := range names {
		top := s.E.S.TopPinsInTopic(t, 1)
		name := ""
		if len(top) > 0 {
			name = s.E.S.Pin(top[0]).Topic
		}
		names[t] = map[string]any{"id": t, "name": name}
	}
	return names, nil
}
