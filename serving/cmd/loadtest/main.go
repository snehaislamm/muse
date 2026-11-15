// Command loadtest drives concurrent home-feed traffic (with interleaved save
// events, so reads race real writes) and reports throughput and tail latency.
//
//	go run ./cmd/loadtest -url http://localhost:8080 -c 16 -d 20s
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sort"
	"sync"
	"time"
)

func main() {
	base := flag.String("url", "http://localhost:8080", "server")
	conc := flag.Int("c", 16, "concurrent clients")
	dur := flag.Duration("d", 20*time.Second, "duration")
	users := flag.Int("users", 4000, "user id range")
	variant := flag.String("variant", "full", "feed variant")
	flag.Parse()

	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *conc}}
	var mu sync.Mutex
	var lat []time.Duration
	var errs, saves int
	deadline := time.Now().Add(*dur)
	var wg sync.WaitGroup
	for w := 0; w < *conc; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			for time.Now().Before(deadline) {
				u := rng.Intn(*users)
				t0 := time.Now()
				resp, err := client.Get(fmt.Sprintf("%s/v1/feed?user=%d&n=25&variant=%s", *base, u, *variant))
				d := time.Since(t0)
				ok := err == nil && resp.StatusCode == 200
				if resp != nil {
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
				saved := false
				if ok && rng.Float64() < 0.1 { // 10% of feed views lead to a save
					body := fmt.Sprintf(`{"user":%d,"pin":%d,"type":"save"}`, u, rng.Intn(20000))
					if r, err := client.Post(*base+"/v1/events", "application/json", bytes.NewBufferString(body)); err == nil {
						io.Copy(io.Discard, r.Body)
						r.Body.Close()
						saved = true
					}
				}
				mu.Lock()
				if ok {
					lat = append(lat, d)
				} else {
					errs++
				}
				if saved {
					saves++
				}
				mu.Unlock()
			}
		}(int64(w))
	}
	wg.Wait()
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	pct := func(p float64) time.Duration {
		if len(lat) == 0 {
			return 0
		}
		return lat[min(len(lat)-1, int(p*float64(len(lat))))]
	}
	fmt.Printf("requests=%d errors=%d saves=%d qps=%.0f p50=%v p95=%v p99=%v max=%v\n",
		len(lat), errs, saves, float64(len(lat))/dur.Seconds(), pct(.5), pct(.95), pct(.99), pct(1))
}
