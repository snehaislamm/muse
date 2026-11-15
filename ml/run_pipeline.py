"""End-to-end offline pipeline: data -> embeddings -> user clusters -> spam ->
ranker -> evaluation -> serving artifacts.

    python run_pipeline.py            # writes ../data and ../artifacts
"""

from __future__ import annotations

import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from muse_ml import datagen, debias, embeddings, features, position_study, ranker, spam

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_f32(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", *arr.shape))
        f.write(arr.tobytes())


def write_i32(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype="<i4")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", *arr.shape))
        f.write(arr.tobytes())


def main(regen: bool = False):
    t0 = time.time()
    if regen or not (DATA / "world.npz").exists():
        log("generating synthetic world")
        datagen.build(DATA)
    z = np.load(DATA / "world.npz")
    st, te, imps, visual = z["saves_train"], z["saves_test"], z["impressions"], z["visual"]
    pins = json.load(open(DATA / "pins_truth.json"))
    boards = json.load(open(DATA / "boards.json"))
    P, U = len(pins), int(st[:, 0].max()) + 1
    topic_ids = np.array([p["topic_id"] for p in pins])
    created = np.array([p["created_day"] for p in pins])
    is_spam = np.array([p["is_spam"] for p in pins], dtype=int)
    metrics: dict = {
        "n_pins": P,
        "n_users": U,
        "n_boards": len(boards),
        "n_saves_train": int(len(st)),
        "n_saves_test": int(len(te)),
        "n_impressions": int(len(imps)),
    }

    # ---- embeddings -------------------------------------------------------
    log("graph embeddings (PPMI + truncated SVD)")
    g = embeddings.graph_embeddings(st, P, len(boards))
    counts = np.bincount(st[:, 2], minlength=P).astype(float)
    W = embeddings.fit_content_projection(visual, g, counts)
    emb = embeddings.unified_embeddings(visual, g, counts, W)
    metrics["cold_pin_frac"] = float((counts == 0).mean())

    # ---- PinnerSage user clusters ----------------------------------------
    log("user interest clusters (Ward)")
    by_user = defaultdict(list)
    for u, _board, p, d in st:
        by_user[int(u)].append((int(p), int(d)))
    clusters: dict[int, list[list[int]]] = {}
    for u in range(U):
        rows = by_user.get(u, [])
        if not rows:
            clusters[u] = []
            continue
        pid = np.array([r[0] for r in rows])
        day = np.array([r[1] for r in rows])
        if len(pid) == 1:
            clusters[u] = [pid.tolist()]
            continue
        from scipy.cluster.hierarchy import fcluster, linkage

        lab = fcluster(linkage(emb[pid], "ward"), 4.0, "distance")
        decay = np.exp(-np.log(2) * (features.REF_DAY - day) / features.HALF_LIFE)
        cl = sorted(np.unique(lab), key=lambda c: -decay[lab == c].sum())[:5]
        clusters[u] = [sorted(set(pid[lab == c].tolist())) for c in cl]
    metrics["avg_clusters_per_user"] = float(np.mean([len(c) for c in clusters.values()]))

    pos, randomized = imps[:, 5], imps[:, 6].astype(bool)
    eval_users = np.random.default_rng(3).random(U) < 0.2
    tr = ~eval_users[imps[:, 0]]

    # propensities are needed early: spam engagement rates are normalised by expected examinations
    rmask = randomized & tr
    a_hat, b_hat, _ = debias.estimate_propensity(imps[rmask, 2], pos[rmask])
    log(f"propensity fit a={a_hat:.2f} b={b_hat:.2f} (true a=2, b=0.8)")

    # ---- spam -------------------------------------------------------------
    log("spam: MinHash-LSH near-dup clusters + logistic regression")
    # engagement rates are normalised by *expected examinations*, not raw impressions,
    # otherwise pins the old policy buried at the bottom look artificially unengaging
    exam_w = debias.power_law(imps[:, 5], a_hat, b_hat)
    Xs, spam_feat_names = spam.spam_features(pins, imps, exam_w)
    clf, labeled = spam.train_spam_model(Xs, is_spam)
    spam_score = clf.predict_proba(Xs)[:, 1]
    # pick the serving filter threshold on the *labeled* sample: lowest score with precision >= 0.95
    ls, ly = spam_score[labeled], is_spam[labeled]
    thr = 0.99
    for t in np.sort(np.unique(ls)):
        sel = ls >= t
        if sel.sum() and ly[sel].mean() >= 0.95:
            thr = float(t)
            break
    ev = ~labeled
    pred = spam_score >= thr
    tp = int((pred & (is_spam == 1) & ev).sum())
    fp = int((pred & (is_spam == 0) & ev).sum())
    fn = int((~pred & (is_spam == 1) & ev).sum())
    metrics["spam"] = {
        "filter_threshold": thr,
        "labeled_frac": float(labeled.mean()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "auc": ranker.auc(is_spam[ev], spam_score[ev]),
        "features": spam_feat_names,
    }

    # ---- ranker -----------------------------------------------------------
    log("ranker features")
    static = features.pin_static(counts, created, spam_score)
    states = {}
    for u in range(U):
        rows = by_user.get(u, [])
        states[u] = features.UserState([r[0] for r in rows], [r[1] for r in rows], clusters[u], emb, visual, topic_ids)
    X = position_study.featurize_impressions(imps, states, emb, visual, topic_ids, static)

    log(f"position-bias study: training naive / IPW / IPW-oracle rankers on {tr.sum()} impressions")
    models, study = position_study.run(X, imps, tr, datagen.load_world(DATA), datagen.examine_prob)
    metrics.update(study)
    model, model_naive = models["ipw"], models["naive"]
    metrics["feature_weight_norms"] = {
        name: dict(zip(features.FEATURES, np.linalg.norm(models[name].W1, axis=1).round(3).tolist(), strict=True))
        for name in ("naive", "ipw")
    }

    # ---- exports for the Go service ------------------------------------
    log("writing artifacts")
    ART.mkdir(exist_ok=True)
    write_f32(ART / "emb.f32", emb)
    write_f32(ART / "visual.f32", visual)
    write_i32(ART / "saves_train.i32", st)
    write_i32(ART / "saves_test.i32", te)
    public = [
        {
            "id": p["id"],
            "title": p["title"],
            "topic": p["topic"],
            "topic_id": p["topic_id"],
            "creator": p["creator"],
            "domain": p["domain"],
            "created_day": p["created_day"],
            "aspect": p["aspect"],
            "hue": p["hue"],
            "tone": p["tone"],
            "spam_score": round(float(spam_score[i]), 6),
            "save_count": int(counts[i]),
        }
        for i, p in enumerate(pins)
    ]
    json.dump(public, open(ART / "pins.json", "w"))
    json.dump([{"id": b["id"], "user": b["user"], "name": b["name"]} for b in boards], open(ART / "boards.json", "w"))
    json.dump(
        {"spam_filter_threshold": thr, "ref_day": features.REF_DAY, "half_life": features.HALF_LIFE},
        open(ART / "serving_config.json", "w"),
    )
    json.dump({str(u): c for u, c in clusters.items()}, open(ART / "user_clusters.json", "w"))
    (ART / "ranker.json").write_text(model.to_json(features.FEATURES))  # served by default (IPW)
    (ART / "ranker_naive.json").write_text(model_naive.to_json(features.FEATURES))
    # training/serving parity fixture: Go must reproduce these numbers
    rng = np.random.default_rng(11)
    fx = []
    for u in rng.choice(U, 25, replace=False):
        pid = rng.choice(P, 8, replace=False)
        f = features.featurize(states[int(u)], pid, emb, visual, topic_ids, static)
        fx.append({"user": int(u), "pins": pid.tolist(), "features": f.tolist(), "pred": model.predict(f).tolist()})
    json.dump(fx, open(ART / "parity_fixture.json", "w"))
    metrics["pipeline_seconds"] = round(time.time() - t0, 1)
    json.dump(metrics, open(ART / "offline_metrics.json", "w"), indent=2)
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(regen="--regen" in sys.argv)
