import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Board, Debug, Item, Pin, pinBackground, Topic, TONE_SWATCHES } from "./api";
import { columnsFor, masonry } from "./layout";

const GAP = 16;
const VARIANTS = ["full", "full_naive", "ranker", "baseline"];

function useWidth(ref: React.RefObject<HTMLDivElement | null>): number {
  const [w, setW] = useState(1200);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

function PinCard(props: { pin: Item | Pin; width: number; saved: boolean; debug: boolean;
  onOpen: () => void; onSave: () => void; onHide?: () => void }) {
  const { pin, width, saved, debug } = props;
  const item = pin as Item;
  return (
    <div className="pin" style={{ width }}>
      <div className="pin-img" role="button" tabIndex={0} aria-label={`Open ${pin.title}`}
        style={{ height: Math.round(width * pin.aspect), background: pinBackground(pin) }}
        onClick={props.onOpen} onKeyDown={(e) => e.key === "Enter" && props.onOpen()}>
        <span className="pin-topic">{pin.topic.replace("_", " ")}</span>
        <button className={`save ${saved ? "saved" : ""}`} aria-pressed={saved}
          onClick={(e) => { e.stopPropagation(); props.onSave(); }}>{saved ? "Saved" : "Save"}</button>
        {props.onHide && (
          <button className="hide" aria-label="Hide pin" title="See fewer like this"
            onClick={(e) => { e.stopPropagation(); props.onHide!(); }}>×</button>
        )}
        {pin.tone > 0 && <span className="tone-dot" title={`Skin tone range ${pin.tone}`}
          style={{ background: TONE_SWATCHES[pin.tone - 1] }} />}
      </div>
      <div className="pin-meta">
        <div className="pin-title">{pin.title}</div>
        {debug && item.sources && (
          <div className="pin-debug">
            {item.sources.map((s) => <span key={s} className={`src src-${s}`}>{s}</span>)}
            <span className="psave">p(save) {item.p_save.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Grid(props: { items: (Item | Pin)[]; saved: Set<number>; debug: boolean; onOpen: (p: Pin) => void;
  onSave: (p: Pin) => void; onHide?: (p: Pin) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const cols = columnsFor(width, 220, GAP);
  const colW = Math.floor((width - GAP * (cols - 1)) / cols);
  const { placed, height } = useMemo(() => masonry(props.items, cols, colW, GAP), [props.items, cols, colW]);
  return (
    <div ref={ref} className="grid" style={{ height }}>
      {placed.map(({ item, column, top }) => (
        <div key={item.id} className="cell" style={{ transform: `translate(${column * (colW + GAP)}px, ${top}px)` }}>
          <PinCard pin={item} width={colW} saved={props.saved.has(item.id)} debug={props.debug}
            onOpen={() => props.onOpen(item)} onSave={() => props.onSave(item)}
            onHide={props.onHide ? () => props.onHide!(item) : undefined} />
        </div>
      ))}
    </div>
  );
}

function DebugPanel({ d }: { d: Debug }) {
  return (
    <aside className="debug" aria-label="Ranking debug panel">
      <strong>{d.variant}</strong> · {d.total_candidates} candidates
      {d.candidates && Object.entries(d.candidates).map(([k, v]) => <span key={k} className={`src src-${k}`}>{k} {v}</span>)}
      <span>total {d.stage_ms.total?.toFixed(1)} ms</span>
      {["retrieval", "ranking", "rerank"].filter((k) => k in d.stage_ms).map((k) => <span key={k}>{k} {d.stage_ms[k].toFixed(1)}</span>)}
      <span>ILS {d.intra_list_similarity.toFixed(2)}</span>
      {d.tone_entropy !== undefined && <span>tone entropy {d.tone_entropy.toFixed(2)}</span>}
      {d.cold_start_user && <span className="cold">cold-start</span>}
    </aside>
  );
}

function Onboarding({ onDone }: { onDone: (user: number) => void }) {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  useEffect(() => { api.topics().then(setTopics).catch(console.error); }, []);
  const toggle = (id: number) => setPicked((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  return (
    <div className="modal-bg">
      <div className="modal onboarding" role="dialog" aria-label="Pick your interests">
        <h2>What are you into?</h2>
        <p>Pick 3 or more so your first feed already feels like yours.</p>
        <div className="chips">
          {topics.map((t) => (
            <button key={t.id} className={`chip ${picked.has(t.id) ? "on" : ""}`} aria-pressed={picked.has(t.id)}
              onClick={() => toggle(t.id)}>{t.name.replace("_", " ")}</button>
          ))}
        </div>
        <button className="primary" disabled={picked.size < 3}
          onClick={() => api.signup([...picked]).then((r) => onDone(r.user))}>Continue</button>
      </div>
    </div>
  );
}

export function App() {
  const [user, setUser] = useState(12);
  const [variant, setVariant] = useState("full");
  const [tone, setTone] = useState(0);
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [debugInfo, setDebugInfo] = useState<Debug | null>(null);
  const [showDebug, setShowDebug] = useState(true);
  const [n, setN] = useState(50);
  const [saved, setSaved] = useState<Set<number>>(new Set());
  const [open, setOpen] = useState<Pin | null>(null);
  const [related, setRelated] = useState<Item[]>([]);
  const [view, setView] = useState<"home" | "boards">("home");
  const [boards, setBoards] = useState<Board[]>([]);
  const [onboarding, setOnboarding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sentinel = useRef<HTMLDivElement>(null);

  const load = useCallback(() => {
    const p = activeQuery ? api.search(activeQuery, user, tone) : api.feed(user, n, variant, tone);
    p.then((r) => { setItems(r.items); setDebugInfo(r.debug); setError(null); }).catch((e) => setError(String(e)));
  }, [user, n, variant, tone, activeQuery]);

  useEffect(load, [load]);
  useEffect(() => { setN(50); setSaved(new Set()); }, [user, variant]);
  useEffect(() => { if (view === "boards") api.boards(user).then(setBoards).catch((e) => setError(String(e))); }, [view, user, saved]);
  useEffect(() => { if (open) api.related(open.id, user).then((r) => setRelated(r.items)).catch(console.error); }, [open, user]);
  useEffect(() => { // infinite scroll: grow the slate when the sentinel is visible
    const el = sentinel.current;
    if (!el || activeQuery) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) setN((x) => Math.min(x + 25, 200)); });
    io.observe(el);
    return () => io.disconnect();
  }, [activeQuery, items.length]);

  const save = (p: Pin) => {
    if (saved.has(p.id)) return;
    api.save(user, p.id).then(() => setSaved((s) => new Set(s).add(p.id))).catch((e) => setError(String(e)));
  };
  const hide = (p: Pin) => {
    setItems((xs) => xs.filter((x) => x.id !== p.id)); // optimistic
    api.hide(user, p.id).catch((e) => setError(String(e)));
  };
  const openPin = (p: Pin) => { setOpen(p); api.click(user, p.id).catch(() => undefined); };

  return (
    <div className="app">
      <header>
        <div className="logo" aria-label="Muse">M</div>
        <nav>
          <button className={view === "home" ? "tab on" : "tab"} onClick={() => { setView("home"); setActiveQuery(""); }}>Home</button>
          <button className={view === "boards" ? "tab on" : "tab"} onClick={() => setView("boards")}>Boards</button>
        </nav>
        <form className="search" onSubmit={(e) => { e.preventDefault(); setView("home"); setActiveQuery(query.trim()); }}>
          <input aria-label="Search" placeholder="Search for ideas" value={query} onChange={(e) => setQuery(e.target.value)} />
        </form>
        <div className="tones" role="group" aria-label="Skin tone range filter">
          {TONE_SWATCHES.map((c, i) => (
            <button key={c} className={`tone ${tone === i + 1 ? "on" : ""}`} style={{ background: c }}
              aria-label={`Skin tone range ${i + 1}`} aria-pressed={tone === i + 1}
              onClick={() => setTone(tone === i + 1 ? 0 : i + 1)} />
          ))}
        </div>
        <label className="ctl">user <input type="number" min={0} value={user} onChange={(e) => setUser(Number(e.target.value) || 0)} /></label>
        <label className="ctl">variant
          <select value={variant} onChange={(e) => setVariant(e.target.value)}>{VARIANTS.map((v) => <option key={v}>{v}</option>)}</select>
        </label>
        <label className="ctl"><input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} /> debug</label>
        <button className="primary" onClick={() => setOnboarding(true)}>New user</button>
      </header>
      {error && <div className="error" role="alert">{error}</div>}
      {showDebug && debugInfo && view === "home" && <DebugPanel d={debugInfo} />}
      <main>
        {view === "home" && (
          <>
            {activeQuery && <h2 className="results-for">Results for “{activeQuery}”</h2>}
            <Grid items={items} saved={saved} debug={showDebug} onOpen={openPin} onSave={save} onHide={hide} />
            <div ref={sentinel} className="sentinel" />
          </>
        )}
        {view === "boards" && (
          <div className="boards">
            {boards.length === 0 && <p>No boards yet — save a pin to start one.</p>}
            {boards.map((b) => (
              <div key={b.id} className="board">
                <div className="board-cover">
                  {b.pins.slice(0, 3).map((p) => <div key={p.id} style={{ background: pinBackground(p) }} />)}
                </div>
                <div className="board-name">{b.name}</div>
                <div className="board-count">{b.pins.length} pins</div>
              </div>
            ))}
          </div>
        )}
      </main>
      {open && (
        <div className="modal-bg" onClick={() => setOpen(null)}>
          <div className="modal closeup" role="dialog" aria-label={open.title} onClick={(e) => e.stopPropagation()}>
            <div className="closeup-top">
              <div className="closeup-img" style={{ background: pinBackground(open), aspectRatio: `1 / ${open.aspect}` }} />
              <div className="closeup-info">
                <button className="close" aria-label="Close" onClick={() => setOpen(null)}>×</button>
                <h2>{open.title}</h2>
                <p className="muted">{open.domain} · {open.topic.replace("_", " ")} · {open.save_count} saves</p>
                <button className="primary" onClick={() => save(open)}>{saved.has(open.id) ? "Saved" : "Save"}</button>
              </div>
            </div>
            <h3>More ideas like this</h3>
            <Grid items={related} saved={saved} debug={false} onOpen={openPin} onSave={save} />
          </div>
        </div>
      )}
      {onboarding && <Onboarding onDone={(u) => { setOnboarding(false); setUser(u); setView("home"); setActiveQuery(""); setQuery(""); setTone(0); }} />}
    </div>
  );
}
