# Detailed results

All numbers are measured on simulated data and reproduced with `make data eval robustness ab loadtest`.
See [README](../README.md) for context and [DESIGN.md](DESIGN.md) for the reasoning behind each choice.

## Retrieval, ranking, serving

**Retrieval (800 users, held-out saves from days 86-89)**

| Source | Recall of held-out saves | Recall on cold pins (0 train saves) |
|---|---|---|
| HNSW user-mean (200) | 0.145 | 0.025 |
| Pixie walk (150) | 0.091 | 0.000 |
| Fresh-pin index (60) | 0.143 | **0.209** |
| **Union of all sources (410)** | **0.336** (2.3x the single source) | **0.210** (8x) |

**Spam:** AUC 0.966. At the serving threshold (picked for ≥95% precision on the reviewed sample),
held-out precision is 0.925 and recall 0.540. Half the spam reuses legitimate titles, so title
de-duplication alone is not enough. Engagement-rate features are normalised by *expected
examinations* rather than raw impressions (see position bias below).

## Position-bias correction (the headline experiment)

The logged impressions come from an older popularity-first feed, and Pinners only act on pins
they actually look at. The hidden examination curve is p(pos) = (1 + pos/2)^-0.8: slot 0 is
looked at about 8x more often than slot 24. A ranker trained on raw logs therefore learns
"popular = good", because popular pins were *shown higher*, not because they are better.

1. **Estimating the bias.** 10% of logged sessions were shown in random order. Fitting a power
   law to click rate by slot on those 23.9k impressions recovers the curve to within 0.013 at
   every slot (estimated a=1.70, b=0.74 vs true 2, 0.8).
2. **IPW training.** Each label is divided by the propensity of the slot it was logged in, so the
   sigmoid/BCE gradient is unbiased for true relevance. A model trained with the *true*
   propensities is indistinguishable from one trained with the estimated ones (1.461 vs 1.461
   expected saves).
3. **Do offline metrics agree with reality?** I scored 7 rankers on 2,245 held-out logged sessions
   with three estimators of "expected saves if this order were shown" and compared each with the
   hidden ground truth:

| Estimator | Rank agreement with truth (Kendall τ, 7 rankers) | Predicted lift, IPW vs naive ranker |
|---|---|---|
| Naive (logged saves as labels) | 0.52 | **−2.9%** (wrong sign) |
| Naive nDCG@10 | 0.52 | −4.5% (wrong sign) |
| **IPS (saves ÷ logged propensity)** | **0.90** | **+3.3%** |
| Ground truth | 1.00 | +2.1% |
| **Online A/B (4,000 simulated users)** | | **+14.1%** [+11.4, +17.1], p < 1e-22 |

The naive offline metric would have **killed the better model**. It also ranks the old logging
policy above embedding similarity, which the truth reverses. The IPW model even has a *lower* save
AUC on logged labels (0.749 vs 0.765), another way raw-log metrics reward the bias. In the online
test, IPW also cut hides 21% [−33, −6], raised saves on fresh pins 4.7x, and raised
under-represented-tone users' saves on tone-bearing pins 34% [+12, +64].

**Why the online lift is 4x the offline estimate:** the logged candidate sets came from a
popularity-first policy and contain few fresh, unpopular pins, which is exactly where the naive
model does the most damage. Counterfactual evaluation is only as good as the logging policy's
coverage. IPS gets the *direction* and the model ranking right; it cannot see items the old
system never showed.

### Robustness: does the conclusion depend on the simulator's assumptions?

`experiments/position_robustness.py` regenerates the logs under five hidden settings, including one
where the true examination curve is **not** in the power-law family the propensity model fits, and
reruns the identical study code:

| Hidden setting | Propensity max error | True lift (IPW vs naive) | Naive metric predicts | IPS predicts | τ naive | τ IPS |
|---|---|---|---|---|---|---|
| weak bias (b=0.4) | 0.055 | +0.5% | −0.9% | +0.4% | 0.71 | 0.90 |
| default (b=0.8) | 0.056 | +2.4% | −7.5% | +1.5% | 0.33 | 0.90 |
| strong bias (b=1.4) | 0.024 | +4.9% | −4.5% | +9.4% | 0.05 | 0.81 |
| misspecified: exponential decay | 0.185 | +3.7% | −2.9% | +3.2% | −0.14 | 0.71 |
| little exploration (2% randomized) | 0.084 | +3.6% | −2.1% | +4.0% | 0.43 | 0.81 |

In every setting the naive offline metric points the wrong way and IPS points the right way. The
gain from IPW grows with the strength of the bias. Misspecification costs accuracy in the
propensity estimate (max error 0.185) but not the decision. These runs use a different random seed
for the logs than the main pipeline, which is why the default row differs slightly from the
headline table.

**ANN (20k × 64-d real embeddings):** HNSW recall@100 = 0.9975 at 0.20 ms/query, vs 4.0 ms
for exact search (20x).

**Online A/B (simulated Pinners, 3 sessions each, same position-bias model, saves/hides fed back live)**

| Experiment | Users | Saves / user | Fresh-pin saves | Hides |
|---|---|---|---|---|
| naive ranker → IPW ranker (both "full") | 4,000 | **+14.1%** [+11.4, +17.1] | **4.7x** | **−21%** [−33, −6] |
| baseline → ranker | 2,000 | **+17.3%** [+12.4, +22.1] | −67% | n.s. |
| baseline → full | 2,000 | **+18.8%** [+14.2, +23.6], p < 1e-17 | −60% | n.s. |
| ranker → full (adds fresh source + re-ranking) | 4,000 | −1.0% [−3.4, +1.4] (n.s.) | +13% (n.s.) | n.s. |

Adding the fresh source and the re-ranking layer is **save-neutral**. Offline, it cut intra-list
similarity from 0.800 to 0.764 and raised tone-range entropy from 0.80 to 0.94. Even the debiased
ranker still gives fresh pins about a third of the saves that the similarity-only baseline does:
popularity is *genuinely* predictive among older pins, so a brand-new pin with 0 saves looks like a
bad old pin. IPW fixes the position confound but not this cold-start one (see next steps).

**Serving:** full funnel (≈410 candidates → 25 pins) averages 5.5 ms. Load test on a 2-vCPU box with
10% of requests followed by a save: **359 QPS, p50 5.1 ms, p99 10.7 ms, 0 errors**. A profile-guided
pass (dense scratch arrays replacing hash maps in Pixie, incremental MMR) cut mean "full" latency from 31 ms to 6 ms.
