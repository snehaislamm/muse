// Typed client for the Muse Go API.

export interface Pin {
  id: number;
  title: string;
  topic: string;
  topic_id: number;
  creator: number;
  domain: string;
  created_day: number;
  aspect: number;
  hue: number;
  tone: number;
  spam_score: number;
  save_count: number;
}

export interface Item extends Pin {
  score: number;
  p_save: number;
  p_click: number;
  p_hide: number;
  sources: string[] | null;
}

export interface Debug {
  variant: string;
  candidates: Record<string, number> | null;
  total_candidates: number;
  stage_ms: Record<string, number>;
  tone_entropy?: number;
  intra_list_similarity: number;
  cold_start_user: boolean;
}

export interface FeedResponse {
  items: Item[];
  debug: Debug;
}

export interface Board {
  id: number;
  user: number;
  name: string;
  pins: Pin[];
}

export interface Topic {
  id: number;
  name: string;
}

const BASE: string = (globalThis as { MUSE_API?: string }).MUSE_API ?? "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = ((await r.json()) as { error?: string }).error ?? msg;
    } catch { /* non-JSON error body */ }
    throw new Error(`${r.status}: ${msg}`);
  }
  return (await r.json()) as T;
}

const post = (path: string, body: unknown) =>
  req<Record<string, unknown>>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  feed: (user: number, n: number, variant: string, tone: number) =>
    req<FeedResponse>(`/v1/feed?user=${user}&n=${n}&variant=${variant}&tone=${tone}`),
  related: (pin: number, user: number) => req<FeedResponse>(`/v1/pins/${pin}/related?user=${user}&n=24`),
  search: (q: string, user: number, tone: number) =>
    req<FeedResponse>(`/v1/search?q=${encodeURIComponent(q)}&user=${user}&tone=${tone}&n=60`),
  boards: (user: number) => req<Board[]>(`/v1/users/${user}/boards`),
  topics: () => req<Topic[]>(`/v1/topics`),
  save: (user: number, pin: number) => post("/v1/events", { user, pin, type: "save" }),
  hide: (user: number, pin: number) => post("/v1/events", { user, pin, type: "hide" }),
  click: (user: number, pin: number) => post("/v1/events", { user, pin, type: "click" }),
  signup: (topics: number[]) => post("/v1/users", { topics }) as Promise<{ user: number }>,
};

// Placeholder artwork: a deterministic gradient per pin (the synthetic corpus has no images).
export function pinBackground(p: Pin): string {
  const h2 = (p.hue + 40) % 360;
  return `linear-gradient(${(p.id * 37) % 360}deg, hsl(${p.hue} 55% 62%), hsl(${h2} 60% 38%))`;
}

export const TONE_SWATCHES = ["#f3d3b5", "#d9a77c", "#a8714a", "#5f3b25"];
