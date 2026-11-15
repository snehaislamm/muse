// Masonry layout: place each pin in the currently shortest column.
// Pure function so it can be unit-tested without a DOM.

export interface Sized {
  id: number;
  aspect: number; // height / width
}

export interface Placed<T extends Sized> {
  item: T;
  column: number;
  top: number;
  height: number;
}

export function masonry<T extends Sized>(items: T[], columns: number, colWidth: number, gap: number,
  footer = 56): { placed: Placed<T>[]; height: number } {
  const cols = Math.max(1, Math.floor(columns));
  const heights: number[] = new Array(cols).fill(0);
  const placed: Placed<T>[] = [];
  for (const item of items) {
    let c = 0;
    for (let i = 1; i < cols; i++) if (heights[i] < heights[c]) c = i; // leftmost shortest column
    const h = Math.round(colWidth * item.aspect) + footer;
    placed.push({ item, column: c, top: heights[c], height: h });
    heights[c] += h + gap;
  }
  return { placed, height: Math.max(0, ...heights) };
}

export function columnsFor(width: number, minCol = 236, gap = 16): number {
  return Math.max(1, Math.floor((width + gap) / (minCol + gap)));
}
