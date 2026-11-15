# Design notes

## 1. Why a synthetic world with a hidden truth
Real engagement logs are private, and offline metrics on logged data are biased by the policy that
produced them. `datagen.py` generates 20k pins (16 topics × 4 styles, 4% spam), 4k users with 1-4
weighted interests, ~14k boards and ~283k saves. Saves are driven by a hidden affinity model with
preferential attachment, so popularity ends up heavy-tailed. Users with a tone-range preference save
matching beauty and fashion pins more often, and the corpus tone distribution is deliberately skewed
55/25/12/8. Saves on or after day 86 are held out, and 27% of those land on pins with **zero**
training saves. That share is what makes freshness a real problem here, not a hypothetical one.

## 2. Retrieval
* **Graph embedding**: PPMI-weighted pin×board matrix → truncated SVD (32-d). Context smoothing
  (board^0.75) stops very large boards from dominating.
* **Cold start**: ridge regression from the visual embedding to the graph space, fitted on pins with ≥3
  saves. It is blended in by confidence n/(n+3). The unified vector is [graph ; 0.5·visual], L2-normalised.
* **Negative result: multi-interest retrieval.** I expected PinnerSage-style per-cluster retrieval to
  beat a single decayed mean vector for users with 3+ interests. Brute-force evaluation said otherwise.
  For those users recall@100 was 0.055 for the mean vs 0.046-0.051 for max-over-centroids or
  importance-allocated retrieval, at 3 Ward distance thresholds. For other users it was 0.129 vs
  0.090-0.125, and medoids did worse than centroids. My explanation is that the topics are nearly orthogonal in 64-d, so the mean
  vector's dot product already decomposes into per-interest similarities. I kept per-interest
  retrieval as an extra *source*, because it adds distinct candidates, but the main source is the mean.
* **Fresh source**: a separate HNSW over pins created in the last 7 days. Cold-pin recall goes from
  0.025 to 0.210. Online, though, the ranker demotes most of these candidates (see §3.1, cold-start conflation).
* **Pixie**: walks with restart (α = 0.5) from up to 20 recent saves. Steps are allocated by
  w·deg·(Cmax − log deg) so viral pins do not take the whole budget. A walk stops once 600 pins have
  ≥4 visits, and scores combine as (Σ√visits)².

## 3. Ranking
The multi-task MLP (7 → 32 ReLU → 3 sigmoids) is written from scratch in NumPy (forward pass,
backprop, Adam) and re-implemented in Go, so serving does not depend on an ML runtime. The save head
gets 2× loss weight. The features are defined once in `features.py`. Heads are combined as
utility = save + 0.3·click − 2·hide, because clickbait spam has high p(click) but high p(hide).

### 3.1 Position bias (`debias.py`)
* **Data.** The logging policy sorts each 25-pin session by popularity (plus Gumbel noise), and users
  examine slot k with probability (1 + k/2)^-0.8. Labels exist only for examined pins. 10% of sessions
  are shown in uniformly random order; this costs engagement in a real product, so it is kept small.
* **Propensity estimation.** On the randomized slice, relevance is independent of slot, so
  CTR(k) ∝ p(k). A grid-search maximum-likelihood fit of (1 + k/a)^-b gives a=1.70, b=0.74, with a
  max error of 0.013 vs the truth. I chose a parametric fit over per-slot CTRs because slots 20-24
  have only ~1k impressions each.
* **IPW.** Targets become y / max(p(k), 0.1). With a sigmoid output the BCE gradient is (s − y/p),
  whose expectation over examination equals (s − relevance). Targets above 1 are fine for this
  gradient, and the 0.1 clip bounds the variance (never active here, since p(24) = 0.13).
* **What the bias did to the naive model.** Holding other features at their means, the naive
  model's utility rises 3.8x as a pin goes from 0 to ~19 saves (0.030 → 0.114). The IPW model's
  rises 1.5x (0.210 → 0.314).
* **Offline estimators.** For a candidate order, expected saves = Σ_i rel_i · p(new slot of i).
  Using logged saves as rel_i ("naive") credits whatever the old policy put on top. Using saves /
  p(logged slot) (IPS) is unbiased. Across 7 rankers, Kendall τ vs the ground-truth ordering is
  0.52 (naive) vs 0.90 (IPS). Naive predicts −2.9% for IPW-vs-naive, IPS predicts +3.3%, the
  truth is +2.1%, and online it is +14.1%.
* **Why online ≫ offline.** Logged candidate sets come from the popular-first policy, so fresh,
  unpopular pins are rare in them, and those are exactly the pins the naive model buries (fresh-pin
  saves online: 0.047 → 0.22 per user). An unbiased estimator still only covers the actions the old
  policy took. To evaluate a policy that shows new kinds of items you need exploration traffic, or
  an online test.
* **Side effects.** Spam features that use engagement rates are normalised by expected examinations
  (Σ p(k)), not raw impressions. Position bias still cost the spam model recall: 0.77 at 95% precision before,
  0.54 at 92% now, because fewer hides are observed. Under-represented-tone users
  saved 34% more tone-bearing pins under IPW, because the naive model's popularity bias favoured the
  majority range.

* **Robustness.** `position_study.run` is shared by the pipeline and `experiments/position_robustness.py`,
  so the sweep runs the same code as the headline result. The sign of every conclusion held under weak
  and strong bias, an exponential curve the power-law fit cannot represent, and a 2% exploration
  budget (table in RESULTS.md). The size of the gain did not hold: it ranges from +0.5% to +4.9%.

### 3.2 Cold-start conflation (open problem)
Even the IPW ranker gives fresh pins about a third of the saves that similarity-only ranking does.
Among older pins, save count is genuinely predictive of quality, so a 3-day-old pin with 0 saves
scores like a bad 80-day-old pin. Position-bias correction cannot fix this, because it is a feature
confound, not a logging one. Candidates: popularity × age interaction features, dropping popularity
for pins under 7 days old, and an exploration slot.

**Parity**: `run_pipeline.py` writes 200 (user, pin) feature vectors and predictions, and
`TestParityWithPython` recomputes them in Go. `pins.json` stores spam scores rounded to 6 decimals, so the
tolerance (1e-4) is set to accept that rounding and nothing larger. Any change to the feature
definitions on one side fails CI.

## 4. Re-ranking and responsible-AI choices
* Filters: spam score above the reviewed threshold, hidden pins, already-saved pins.
* MMR with incremental max-similarity (O(n·k·d)), replacing a naive O(n·k²·d) loop.
* Creator cap: at most 2 pins per creator per slate.
* Tone diversity: a penalty proportional to a range's over-representation (vs 1/4) among tone-bearing
  pins already in the slate, plus an **opt-in** filter in search and feed. I chose not to infer any
  user's skin tone from behaviour. The system diversifies by default and lets people choose.
* Result (ranker → full, which also adds the fresh source; 4,000 simulated users): saves −1.0%
  [−3.4, +1.4], not significant. Offline, ILS went from 0.800 to 0.764 and tone entropy from 0.80 to 0.94.
  On the v1 dataset (before position bias was modelled) the same comparison gave −3.9% and then
  −1.4% in a replication, and a six-setting λ/cap/tone sweep stayed within ±1.4%
  (`experiments/results/v1_rerank_sweep_raw.txt`). My conclusion is that this layer buys diversity
  for a save cost of a few percent at most.

## 5. Serving
* Store: a single RWMutex. Reads copy small slices, and Pixie walks the adjacency lists in place
  under the read lock (`WithGraph`). Events take the write lock, and each new save joins the nearest
  interest cluster (cos > 0.5) or starts a new one, so the next request is already personalised.
* HNSW: the neighbour-selection heuristic from Malkov & Yashunin (Alg. 4), and a pooled
  generation-stamped visited array instead of a per-search map. Recall on hard clustered synthetic data
  rises monotonically with ef (0.83 → 0.998 for ef 50 → 400), which I used as a correctness check.
* Profiling (pprof) showed Pixie walks using about 60% of request CPU, much of it in hash-map operations. Switching to pooled dense
  arrays with touched-lists, plus incremental MMR, brought the mean "full" request from 31 ms to about 6 ms.
* Operability: structured JSON logs, panic recovery, input validation with 4xx errors, request and
  write timeouts, graceful shutdown, Prometheus histograms per endpoint, and counters per candidate
  source. Tests run under the race detector.

## 6. Experimentation
Users are assigned by SHA-256(salt:user) mod 2. Each experiment uses a fresh server, or disjoint user
ranges, so online state does not leak between arms. Welch's t-test and a 2,000-resample bootstrap CI
give the lift, with hides and server latency tracked as guardrails. Override query parameters
(`lambda`, `cap`, `tonediv`) let one binary serve a parameter sweep, and the `full_naive` variant serves
the uncorrected ranker next to the IPW one. The simulated users examine slots with the same
position-bias curve as the logs.
