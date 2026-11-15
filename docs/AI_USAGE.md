# How AI tools were used on this project, and how their output was checked

I built Muse with an AI coding assistant (Claude) as a pair programmer. It helped scaffold modules,
draft the HNSW and Pixie implementations from the papers, and write tests. I treated every output as
unverified until a test, a measurement or the ground truth confirmed it.

| Risk with AI-generated code | How it was validated here |
|---|---|
| Plausible but wrong algorithm (e.g. a subtle HNSW bug) | recall vs exact brute force on clustered data. Recall must rise monotonically with ef, and it does (0.83 → 0.998) |
| Python and Go silently disagreeing | parity fixture test, 1e-4 tolerance (observed 5e-7) |
| Concurrency bugs in shared state | `go test -race`, a concurrent insert/search test, and a load test that interleaves saves with feed reads |
| Tests that only confirm the author's assumptions | hand-computed expectations (AUC vs brute-force pair counting, masonry positions). Two AI-drafted tests had wrong expected values and were corrected after checking the math |
| "Improvements" that don't improve anything | every component is a switchable variant measured offline and in the A/B simulator. Two ideas I expected to win (multi-interest retrieval as the main source; re-ranking boosting saves) did not, and they are documented as negative results |
| Offline metrics that look right but are biased | each offline estimator is checked against the hidden ground truth. The naive logged-label metric picked the wrong ranker; IPS matched the truth (Kendall τ 0.90) |
| Security and robustness | bounded request bodies, validated query params, panic recovery, no user attributes inferred |

Every claim in the README and `DESIGN.md` is backed by a test or a measurement that can be re-run from this repo.
