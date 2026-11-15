// Command server runs the Muse discovery API.
//
//	go run ./cmd/server -artifacts ../artifacts -addr :8080 -static ../web/dist
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/snehaislam/muse/serving/internal/api"
	"github.com/snehaislam/muse/serving/internal/feed"
	"github.com/snehaislam/muse/serving/internal/metrics"
	"github.com/snehaislam/muse/serving/internal/rank"
	"github.com/snehaislam/muse/serving/internal/store"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	dir := flag.String("artifacts", "../artifacts", "artifact directory from ml/run_pipeline.py")
	static := flag.String("static", "", "optional directory with the built web client")
	verbose := flag.Bool("v", false, "debug logging")
	flag.Parse()

	lvl := slog.LevelInfo
	if *verbose {
		lvl = slog.LevelDebug
	}
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))

	t0 := time.Now()
	s, err := store.Load(*dir)
	if err != nil {
		log.Error("loading artifacts", "err", err)
		os.Exit(1)
	}
	m, err := rank.LoadModel(filepath.Join(*dir, "ranker.json"))
	if err != nil {
		log.Error("loading ranker", "err", err)
		os.Exit(1)
	}
	eng := feed.NewEngine(s, m, metrics.New())
	if naive, err := rank.LoadModel(filepath.Join(*dir, "ranker_naive.json")); err == nil {
		eng.Alt["naive"] = naive // served by the "full_naive" variant for position-bias A/B tests
	} else {
		log.Warn("naive ranker not loaded; full_naive variant falls back to default", "err", err)
	}
	log.Info("engine ready", "pins", s.NumPins(), "fresh_index", eng.Fresh.Len(), "startup_ms", time.Since(t0).Milliseconds())

	srv := &http.Server{Addr: *addr, Handler: api.New(eng, log, *static),
		ReadTimeout: 5 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	go func() {
		log.Info("listening", "addr", *addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server", "err", err)
			os.Exit(1)
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	log.Info("shutting down")
	_ = srv.Shutdown(ctx)
}
