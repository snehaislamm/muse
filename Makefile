.PHONY: all data test test-go test-py test-web lint serve web eval ab robustness loadtest clean

all: data web

data:            ## synthetic world + model training + serving artifacts (~40 s)
	cd ml && python3 run_pipeline.py

web:
	cd web && npm install && npm run build

test: lint test-py test-go test-web

lint:
	ruff check --config ml/pyproject.toml ml experiments
	ruff format --check --config ml/pyproject.toml ml experiments
	test -z "$$(gofmt -l serving)"

test-py:
	cd ml && python3 -m pytest -q tests

test-go:
	cd serving && go vet ./... && go test -race ./...

test-web:
	cd web && npm test && npm run typecheck

serve:           ## http://localhost:8080
	cd serving && go run ./cmd/server -artifacts ../artifacts -static ../web/dist

eval:            ## offline replay of held-out saves
	cd serving && go run ./cmd/offline-eval -artifacts ../artifacts -users 800

ab:              ## needs `make serve` running in another terminal
	python3 experiments/ab_sim.py --control baseline --treatment full --users 1200

robustness:      ## position-bias conclusions under 5 simulator settings (~1 min)
	python3 experiments/position_robustness.py

loadtest:
	cd serving && go run ./cmd/loadtest -c 8 -d 20s

clean:
	rm -rf data artifacts web/dist
