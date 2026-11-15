// Package store loads the offline artifacts and holds all mutable serving state
// (the pin-board graph, user saves/hides, popularity counts).
//
// Concurrency: every exported read method takes the read lock; online writes
// (saves, hides, new pins, new users) take the write lock. Readers never see a
// half-applied event.
package store

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
)

type Pin struct {
	ID         int     `json:"id"`
	Title      string  `json:"title"`
	Topic      string  `json:"topic"`
	TopicID    int     `json:"topic_id"`
	Creator    int     `json:"creator"`
	Domain     string  `json:"domain"`
	CreatedDay int     `json:"created_day"`
	Aspect     float64 `json:"aspect"`
	Hue        int     `json:"hue"`
	Tone       int     `json:"tone"`
	SpamScore  float64 `json:"spam_score"`
	SaveCount  int     `json:"save_count"`
}

type Board struct {
	ID   int    `json:"id"`
	User int    `json:"user"`
	Name string `json:"name"`
}

type Save struct{ User, Board, Pin, Day int32 }

type Config struct {
	SpamThreshold float64 `json:"spam_filter_threshold"`
	RefDay        int     `json:"ref_day"`
	HalfLife      float64 `json:"half_life"`
}

// Matrix is a dense row-major float32 matrix.
type Matrix struct {
	Rows, Cols int
	Data       []float32
}

func (m *Matrix) Row(i int) []float32 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

type Store struct {
	mu        sync.RWMutex
	Cfg       Config
	pins      []Pin
	Emb       *Matrix // unified embedding (graph ; visual), L2-normalised
	Visual    *Matrix
	boards    []Board
	boardPins [][]int32
	pinBoards [][]int32
	userSaves map[int][]Save
	clusters  map[int][][]int
	hidden    map[int]map[int]bool
	counts    []float64
	topicPins [][]int // pins per topic sorted by save count (cold-start pools)
	nextUser  int
	TestSaves []Save
}

func readMatrixF32(path string) (*Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hdr [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}
	m := &Matrix{Rows: int(hdr[0]), Cols: int(hdr[1]), Data: make([]float32, int(hdr[0])*int(hdr[1]))}
	return m, binary.Read(f, binary.LittleEndian, m.Data)
}

func readSaves(path string) ([]Save, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hdr [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr[1] != 4 {
		return nil, fmt.Errorf("%s: expected 4 columns, got %d", path, hdr[1])
	}
	raw := make([]int32, hdr[0]*4)
	if err := binary.Read(f, binary.LittleEndian, raw); err != nil && err != io.EOF {
		return nil, err
	}
	out := make([]Save, hdr[0])
	for i := range out {
		out[i] = Save{raw[4*i], raw[4*i+1], raw[4*i+2], raw[4*i+3]}
	}
	return out, nil
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// Load reads every artifact produced by ml/run_pipeline.py.
func Load(dir string) (*Store, error) {
	s := &Store{userSaves: map[int][]Save{}, clusters: map[int][][]int{}, hidden: map[int]map[int]bool{}}
	var err error
	if err = readJSON(filepath.Join(dir, "serving_config.json"), &s.Cfg); err != nil {
		return nil, err
	}
	if err = readJSON(filepath.Join(dir, "pins.json"), &s.pins); err != nil {
		return nil, err
	}
	if err = readJSON(filepath.Join(dir, "boards.json"), &s.boards); err != nil {
		return nil, err
	}
	if s.Emb, err = readMatrixF32(filepath.Join(dir, "emb.f32")); err != nil {
		return nil, err
	}
	if s.Visual, err = readMatrixF32(filepath.Join(dir, "visual.f32")); err != nil {
		return nil, err
	}
	if s.Emb.Rows != len(s.pins) || s.Visual.Rows != len(s.pins) {
		return nil, fmt.Errorf("artifact mismatch: %d pins, %d emb rows, %d visual rows", len(s.pins), s.Emb.Rows, s.Visual.Rows)
	}
	var cl map[string][][]int
	if err = readJSON(filepath.Join(dir, "user_clusters.json"), &cl); err != nil {
		return nil, err
	}
	for k, v := range cl {
		u, _ := strconv.Atoi(k)
		s.clusters[u] = v
		if u >= s.nextUser {
			s.nextUser = u + 1
		}
	}
	saves, err := readSaves(filepath.Join(dir, "saves_train.i32"))
	if err != nil {
		return nil, err
	}
	if s.TestSaves, err = readSaves(filepath.Join(dir, "saves_test.i32")); err != nil {
		return nil, err
	}
	s.boardPins = make([][]int32, len(s.boards))
	s.pinBoards = make([][]int32, len(s.pins))
	s.counts = make([]float64, len(s.pins))
	for _, sv := range saves {
		s.applySave(sv)
	}
	s.rebuildTopicPools()
	return s, nil
}

func (s *Store) applySave(sv Save) {
	s.userSaves[int(sv.User)] = append(s.userSaves[int(sv.User)], sv)
	s.boardPins[sv.Board] = append(s.boardPins[sv.Board], sv.Pin)
	s.pinBoards[sv.Pin] = append(s.pinBoards[sv.Pin], sv.Board)
	s.counts[sv.Pin]++
}

func (s *Store) rebuildTopicPools() {
	maxT := 0
	for _, p := range s.pins {
		if p.TopicID > maxT {
			maxT = p.TopicID
		}
	}
	s.topicPins = make([][]int, maxT+1)
	for _, p := range s.pins {
		if p.SpamScore < s.Cfg.SpamThreshold {
			s.topicPins[p.TopicID] = append(s.topicPins[p.TopicID], p.ID)
		}
	}
	for _, ids := range s.topicPins {
		sort.Slice(ids, func(i, j int) bool { return s.counts[ids[i]] > s.counts[ids[j]] })
	}
}

// ---- read API -------------------------------------------------------------

func (s *Store) NumPins() int   { s.mu.RLock(); defer s.mu.RUnlock(); return len(s.pins) }
func (s *Store) NumTopics() int { s.mu.RLock(); defer s.mu.RUnlock(); return len(s.topicPins) }

func (s *Store) Pin(id int) Pin { s.mu.RLock(); defer s.mu.RUnlock(); return s.pins[id] }

func (s *Store) Count(id int) float64 { s.mu.RLock(); defer s.mu.RUnlock(); return s.counts[id] }

// UserSaves returns a copy of the user's saves in insertion order.
func (s *Store) UserSaves(u int) []Save {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]Save(nil), s.userSaves[u]...)
}

func (s *Store) UserClusters(u int) [][]int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([][]int, len(s.clusters[u]))
	for i, c := range s.clusters[u] {
		out[i] = append([]int(nil), c...)
	}
	return out
}

func (s *Store) Hidden(u int) map[int]bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[int]bool, len(s.hidden[u]))
	for k := range s.hidden[u] {
		out[k] = true
	}
	return out
}

func (s *Store) UserBoards(u int) []Board {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []Board
	for _, b := range s.boards {
		if b.User == u {
			out = append(out, b)
		}
	}
	return out
}

func (s *Store) BoardPins(b int) []int32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]int32(nil), s.boardPins[b]...)
}

// WithGraph runs fn under the read lock with direct access to the bipartite
// adjacency lists (used by the Pixie random walk, which must not copy them).
func (s *Store) WithGraph(fn func(pinBoards, boardPins [][]int32)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.pinBoards, s.boardPins)
}

// TopPinsInTopic returns up to n popular non-spam pins of a topic.
func (s *Store) TopPinsInTopic(t, n int) []int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if t < 0 || t >= len(s.topicPins) {
		return nil
	}
	ids := s.topicPins[t]
	if len(ids) > n {
		ids = ids[:n]
	}
	return append([]int(nil), ids...)
}

// ---- write API (online events) -------------------------------------------

// RecordSave adds a save event. If the user has no board yet a default board is created.
// The new pin joins the nearest interest cluster (cos > 0.5) or starts a new one.
func (s *Store) RecordSave(u, pin, board int) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if pin < 0 || pin >= len(s.pins) {
		return 0, fmt.Errorf("unknown pin %d", pin)
	}
	if board < 0 || board >= len(s.boards) || s.boards[board].User != u {
		board = -1
		for _, b := range s.boards {
			if b.User == u {
				board = b.ID
				break
			}
		}
		if board < 0 {
			board = len(s.boards)
			s.boards = append(s.boards, Board{ID: board, User: u, Name: "Saved ideas"})
			s.boardPins = append(s.boardPins, nil)
		}
	}
	s.applySave(Save{int32(u), int32(board), int32(pin), int32(s.Cfg.RefDay)})
	s.assignCluster(u, pin)
	return board, nil
}

func (s *Store) assignCluster(u, pin int) {
	v := s.Emb.Row(pin)
	best, bestSim := -1, 0.5
	for i, c := range s.clusters[u] {
		cent := make([]float64, s.Emb.Cols)
		for _, p := range c {
			for j, x := range s.Emb.Row(p) {
				cent[j] += float64(x)
			}
		}
		var dot, n float64
		for j := range cent {
			dot += cent[j] * float64(v[j])
			n += cent[j] * cent[j]
		}
		if sim := dot / math.Max(math.Sqrt(n), 1e-9); sim > bestSim {
			best, bestSim = i, sim
		}
	}
	if best >= 0 {
		for _, p := range s.clusters[u][best] {
			if p == pin {
				return
			}
		}
		s.clusters[u][best] = append(s.clusters[u][best], pin)
		return
	}
	if len(s.clusters[u]) < 5 {
		s.clusters[u] = append(s.clusters[u], []int{pin})
	}
}

func (s *Store) RecordHide(u, pin int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hidden[u] == nil {
		s.hidden[u] = map[int]bool{}
	}
	s.hidden[u][pin] = true
}

// NewUser allocates an id for a signed-up user (cold start).
func (s *Store) NewUser() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.nextUser
	s.nextUser++
	return u
}
