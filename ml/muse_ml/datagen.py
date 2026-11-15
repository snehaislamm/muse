"""Synthetic Pinterest-like world: topics, pins, users, boards, saves, impressions.

Everything is generated from a hidden ground-truth model (`World`) so that every
offline metric and every online A/B result can be checked against the truth.
No real user data is used anywhere in this project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

D_LATENT = 32
N_DAYS = 90
TEST_DAY = 86  # saves on/after this day are held out
TONE_TOPICS = {"hair_beauty", "makeup", "womens_fashion", "weddings"}
TONE_DIST = [0.55, 0.25, 0.12, 0.08]  # deliberately skewed, like real corpora

TOPICS: dict[str, dict] = {
    "home_decor": {
        "styles": ["minimalist", "boho", "mid-century", "farmhouse"],
        "nouns": ["living room", "bedroom", "entryway", "shelf styling", "reading nook"],
    },
    "recipes": {
        "styles": ["vegan", "one-pot", "high-protein", "30-minute"],
        "nouns": ["dinner", "pasta", "curry", "meal prep", "soup"],
    },
    "baking": {
        "styles": ["sourdough", "gluten-free", "french", "holiday"],
        "nouns": ["bread", "cookies", "layer cake", "croissants", "pie"],
    },
    "womens_fashion": {
        "styles": ["streetwear", "business casual", "vintage", "capsule"],
        "nouns": ["outfit", "fall look", "work wardrobe", "denim", "dress"],
    },
    "mens_fashion": {
        "styles": ["smart casual", "techwear", "classic", "minimal"],
        "nouns": ["outfit", "jacket", "sneakers", "suit", "layering"],
    },
    "hair_beauty": {
        "styles": ["protective", "curly", "braided", "short"],
        "nouns": ["hairstyle", "hair care routine", "updo", "haircut", "wash day"],
    },
    "makeup": {
        "styles": ["natural", "bold", "editorial", "everyday"],
        "nouns": ["makeup look", "eyeliner", "lip combo", "skincare base", "brows"],
    },
    "weddings": {
        "styles": ["rustic", "garden", "modern", "boho"],
        "nouns": ["bouquet", "table setting", "bridal hair", "venue", "invitation"],
    },
    "travel": {
        "styles": ["budget", "luxury", "backpacking", "road trip"],
        "nouns": ["itinerary", "packing list", "japan guide", "beach escape", "city guide"],
    },
    "diy_crafts": {
        "styles": ["upcycled", "kids", "cricut", "macrame"],
        "nouns": ["project", "wall hanging", "gift idea", "organizer", "planter"],
    },
    "gardening": {
        "styles": ["balcony", "native plants", "vegetable", "indoor"],
        "nouns": ["garden", "raised bed", "plant care", "layout", "herb garden"],
    },
    "fitness": {
        "styles": ["home", "strength", "pilates", "running"],
        "nouns": ["workout", "routine", "plan", "stretch", "core circuit"],
    },
    "art": {
        "styles": ["watercolor", "digital", "ink", "abstract"],
        "nouns": ["painting", "sketch", "illustration", "tutorial", "palette"],
    },
    "architecture": {
        "styles": ["brutalist", "scandinavian", "tiny house", "japandi"],
        "nouns": ["home", "facade", "interior", "cabin", "kitchen"],
    },
    "parenting": {
        "styles": ["montessori", "toddler", "newborn", "rainy day"],
        "nouns": ["activity", "nursery", "routine", "lunchbox", "playroom"],
    },
    "tech": {
        "styles": ["desk setup", "cozy", "rgb", "productivity"],
        "nouns": ["workspace", "keyboard", "home office", "gadget", "cable management"],
    },
}
TOPIC_NAMES = list(TOPICS)
IDEA_WORDS = ["ideas", "inspo", "guide", "for beginners", "aesthetic", "tips", "you'll love", ""]
SPAM_TEMPLATES = [
    "FREE {x} click here $$$",
    "you won't believe this {x} 100% free",
    "{x} limited offer click link",
    "best {x} deal free gift card",
]


@dataclass
class World:
    seed: int
    topic_vecs: np.ndarray  # (T, D)
    style_vecs: np.ndarray  # (T, S, D)
    visual_rot: np.ndarray  # (D, D)
    pins: list[dict] = field(default_factory=list)
    pin_latent: np.ndarray = None  # (P, D) hidden truth
    pin_visual: np.ndarray = None  # (P, D) observable content embedding
    users: list[dict] = field(default_factory=list)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(seed: int = 7, n_pins: int = 20000, n_users: int = 4000, spam_frac: float = 0.04) -> World:
    rng = np.random.default_rng(seed)
    T, S = len(TOPIC_NAMES), 4
    topic_vecs = _unit(rng.normal(size=(T, D_LATENT)))
    style_vecs = _unit(rng.normal(size=(T, S, D_LATENT)))
    q, _ = np.linalg.qr(rng.normal(size=(D_LATENT, D_LATENT)))
    w = World(seed, topic_vecs, style_vecs, q)

    n_spam = int(n_pins * spam_frac)
    n_real = n_pins - n_spam
    n_creators = 900
    creator_topic = rng.integers(0, T, size=n_creators)
    creator_w = 1.0 / np.arange(1, n_creators + 1) ** 0.8
    creator_w /= creator_w.sum()
    latent, visual = [], []
    # popular topics get more content (zipf-ish)
    topic_weights = 1.0 / np.arange(1, T + 1) ** 0.5
    topic_weights /= topic_weights.sum()
    for pid in range(n_pins):
        spam = pid >= n_real
        if spam:
            t = int(rng.integers(0, T))
            s = int(rng.integers(0, S))
            creator = 10_000 + int(rng.integers(0, 120))  # throwaway accounts
            v = _unit(rng.normal(size=D_LATENT))  # off-manifold: relevant to no-one
            noun = rng.choice(TOPICS[TOPIC_NAMES[t]]["nouns"])
            if rng.random() < 0.5:  # templated clickbait
                title = rng.choice(SPAM_TEMPLATES).format(x=noun)
            else:  # stolen title copied from a legit pin: looks clean, near-dups legit content
                src = w.pins[int(rng.integers(0, n_real))]["title"]
                title = src if rng.random() < 0.6 else f"{src} {rng.choice(['shop now', 'see more', 'link in bio'])}"
            domain = f"deals{int(rng.integers(0, 4))}.biz"
            quality = float(rng.beta(1.2, 6))
        else:
            creator = int(rng.choice(n_creators, p=creator_w))  # heavy-tailed creator sizes
            t = int(creator_topic[creator]) if rng.random() < 0.7 else int(rng.choice(T, p=topic_weights))
            s = int(rng.integers(0, S))
            v = _unit(topic_vecs[t] + 0.75 * style_vecs[t, s] + 0.45 * _unit(rng.normal(size=D_LATENT)))
            info = TOPICS[TOPIC_NAMES[t]]
            title = f"{info['styles'][s].capitalize()} {rng.choice(info['nouns'])} {rng.choice(IDEA_WORDS)}".strip()
            domain = (
                f"creator{creator}.com"
                if rng.random() < 0.5
                else rng.choice(["blog.example", "shop.example", "recipes.example", "studio.example"])
            )
            quality = float(rng.beta(4, 3))
        vis = _unit(v @ q + 0.55 * _unit(rng.normal(size=D_LATENT)))
        tone = int(rng.choice(4, p=TONE_DIST)) + 1 if TOPIC_NAMES[t] in TONE_TOPICS and not spam else 0
        w.pins.append(
            {
                "id": pid,
                "title": title,
                "topic": TOPIC_NAMES[t],
                "topic_id": t,
                "style": s,
                "creator": creator,
                "domain": domain,
                "created_day": int(rng.integers(0, N_DAYS)),
                "aspect": round(float(rng.choice([1.0, 1.25, 1.5, 1.78], p=[0.2, 0.3, 0.35, 0.15])), 2),
                "hue": int((t * 22 + s * 9 + rng.integers(0, 12)) % 360),
                "tone": tone,
                "is_spam": bool(spam),
                "_quality": quality,
            }
        )
        latent.append(v)
        visual.append(vis)
    w.pin_latent = np.asarray(latent, dtype=np.float32)
    w.pin_visual = np.asarray(visual, dtype=np.float32)

    for uid in range(n_users):
        k = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.2]))
        topics = rng.choice(T, size=k, replace=False, p=topic_weights)
        weights = rng.dirichlet(np.ones(k) * 1.5)
        interests = []
        for t, wt in zip(topics, weights, strict=True):
            s = int(rng.integers(0, S))
            vec = _unit(topic_vecs[t] + 0.75 * style_vecs[t, s])
            interests.append({"topic_id": int(t), "style": s, "weight": float(wt), "vec": vec})
        tone_pref = int(rng.integers(1, 5)) if rng.random() < 0.6 else 0
        w.users.append(
            {"id": uid, "interests": interests, "tone_pref": tone_pref, "activity": float(rng.lognormal(0, 0.5))}
        )
    return w


def truth_affinity(w: World, uid: int, pin_ids: np.ndarray) -> np.ndarray:
    """Hidden relevance: best-matching interest, weighted by how much the user cares."""
    u = w.users[uid]
    V = w.pin_latent[pin_ids]
    best = np.full(len(pin_ids), -1.0)
    for it in u["interests"]:
        best = np.maximum(best, V @ it["vec"] * (0.75 + 0.5 * it["weight"]))
    return best


def truth_probs(w: World, uid: int, pin_ids: np.ndarray) -> dict[str, np.ndarray]:
    aff = truth_affinity(w, uid, pin_ids)
    qual = np.array([w.pins[p]["_quality"] for p in pin_ids])
    spam = np.array([w.pins[p]["is_spam"] for p in pin_ids], dtype=float)
    tones = np.array([w.pins[p]["tone"] for p in pin_ids])
    pref = w.users[uid]["tone_pref"]
    tone_bonus = np.where((tones > 0) & (pref > 0), np.where(tones == pref, 0.9, -0.4), 0.0)
    p_save = sigmoid(-4.0 + 3.4 * aff + 1.6 * (qual - 0.5) + tone_bonus - 2.0 * spam)
    p_click = sigmoid(-2.2 + 2.0 * aff + 0.8 * (qual - 0.5) + 0.5 * tone_bonus + 1.2 * spam)
    p_hide = sigmoid(-3.5 - 1.5 * aff + 2.8 * spam)
    return {"save": p_save, "click": p_click, "hide": p_hide}


def generate_boards(w: World, rng: np.random.Generator) -> tuple[list[dict], list[tuple]]:
    """Each user curates boards around their interests. Returns boards and saves
    (user, board, pin, day). Preferential attachment makes popularity heavy-tailed."""
    by_topic: dict[int, np.ndarray] = {}
    tid = np.array([p["topic_id"] for p in w.pins])
    spam = np.array([p["is_spam"] for p in w.pins])
    for t in range(len(TOPIC_NAMES)):
        by_topic[t] = np.where((tid == t) | (spam & (rng.random(len(tid)) < 0.02)))[0]
    created = np.array([p["created_day"] for p in w.pins])
    counts = np.zeros(len(w.pins))
    boards, saves = [], []
    for u in w.users:
        for it in u["interests"]:
            n_boards = 1 + int(it["weight"] > 0.45)
            for _ in range(n_boards):
                bid = len(boards)
                info = TOPICS[TOPIC_NAMES[it["topic_id"]]]
                boards.append(
                    {
                        "id": bid,
                        "user": u["id"],
                        "topic_id": it["topic_id"],
                        "name": f"{info['styles'][it['style']]} {TOPIC_NAMES[it['topic_id']].replace('_', ' ')}",
                    }
                )
                cand = by_topic[it["topic_id"]]
                probs = truth_probs(w, u["id"], cand)["save"]
                logits = np.log(probs + 1e-9) * 2.0 + 0.35 * np.log1p(counts[cand])
                pr = np.exp(logits - logits.max())
                pr /= pr.sum()
                n = int(np.clip(rng.poisson(18 * u["activity"]), 6, 60))
                chosen = rng.choice(cand, size=min(n, len(cand)), replace=False, p=pr)
                for p in chosen:
                    day = int(rng.integers(created[p], N_DAYS))
                    saves.append((u["id"], bid, int(p), day))
                    counts[p] += 1
    return boards, saves


def examine_prob(pos):
    """Hidden position-bias model: probability a Pinner even looks at slot `pos`
    of a feed. Clicks, saves and hides are only observed on examined pins."""
    return (1.0 + np.asarray(pos, dtype=float) / 2.0) ** -0.8


def generate_impressions(
    w: World,
    rng: np.random.Generator,
    saves_train: list[tuple],
    sessions_per_user: int = 3,
    per_session: int = 25,
    randomized_frac: float = 0.10,
    exam_fn=None,
) -> np.ndarray:
    """Logged home-feed impressions from an older popularity-first policy.

    The logging policy puts popular pins at the top, and users only act on pins
    they examine (position bias), so popular pins collect labels partly *because*
    of where they were shown. A small slice of sessions (`randomized_frac`) is
    shown in uniformly random order, which is what lets us estimate the bias.

    Rows: user, pin, click, save, hide, position, randomized.
    `exam_fn` overrides the examination curve (used by the robustness sweep).
    """
    exam_fn = exam_fn or examine_prob
    topic_pins = {t: np.where(np.array([p["topic_id"] for p in w.pins]) == t)[0] for t in range(len(TOPIC_NAMES))}
    pop = np.bincount([s[2] for s in saves_train], minlength=len(w.pins)).astype(float)
    pop_p = (pop + 1) / (pop + 1).sum()
    rows = []
    for u in w.users:
        tw = np.array([it["weight"] for it in u["interests"]])
        for _ in range(sessions_per_user):
            n_int = int(per_session * 0.55)
            n_pop = int(per_session * 0.25)
            ts = rng.choice(len(tw), size=n_int, p=tw / tw.sum())
            pins = [int(rng.choice(topic_pins[u["interests"][i]["topic_id"]])) for i in ts]
            pins += list(rng.choice(len(w.pins), size=n_pop, p=pop_p))
            pins += list(rng.integers(0, len(w.pins), size=per_session - len(pins)))
            pins = np.array(pins)
            randomized = rng.random() < randomized_frac
            if randomized:
                order = rng.permutation(len(pins))
            else:  # popularity-first with a little noise
                order = np.argsort(-(np.log1p(pop[pins]) + rng.gumbel(size=len(pins)) * 0.5))
            pins = pins[order]
            pr = truth_probs(w, u["id"], pins)
            seen = rng.random(len(pins)) < exam_fn(np.arange(len(pins)))
            click = seen & (rng.random(len(pins)) < pr["click"])
            save = seen & (rng.random(len(pins)) < pr["save"])
            hide = seen & (rng.random(len(pins)) < pr["hide"]) & ~save
            for i, p in enumerate(pins):
                rows.append((u["id"], int(p), int(click[i]), int(save[i]), int(hide[i]), i, int(randomized)))
    return np.array(rows, dtype=np.int32)


def build(out_dir: Path, seed: int = 7, **kw) -> World:
    rng = np.random.default_rng(seed + 1)
    w = make_world(seed, **kw)
    boards, saves = generate_boards(w, rng)
    train = [s for s in saves if s[3] < TEST_DAY]
    test = [s for s in saves if s[3] >= TEST_DAY]
    imps = generate_impressions(w, rng, train)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "world.npz",
        latent=w.pin_latent,
        visual=w.pin_visual,
        saves_train=np.array(train, dtype=np.int32),
        saves_test=np.array(test, dtype=np.int32),
        impressions=imps,
    )
    with open(out_dir / "boards.json", "w") as f:
        json.dump(boards, f)
    with open(out_dir / "users_truth.json", "w") as f:  # hidden truth, used only by simulators
        json.dump(
            [
                {
                    "id": u["id"],
                    "tone_pref": u["tone_pref"],
                    "interests": [
                        {k: (v.tolist() if k == "vec" else v) for k, v in it.items()} for it in u["interests"]
                    ],
                }
                for u in w.users
            ],
            f,
        )
    with open(out_dir / "pins_truth.json", "w") as f:
        json.dump(w.pins, f)
    return w


def load_world(data_dir: Path) -> World:
    """Rebuild a World (for simulators) from the saved truth files."""
    pins = json.load(open(data_dir / "pins_truth.json"))
    users = json.load(open(data_dir / "users_truth.json"))
    for u in users:
        for it in u["interests"]:
            it["vec"] = np.array(it["vec"])
    z = np.load(data_dir / "world.npz")
    w = World(0, None, None, None, pins=pins, users=users)
    w.pin_latent, w.pin_visual = z["latent"], z["visual"]
    return w
