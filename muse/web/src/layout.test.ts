import { test } from "node:test";
import assert from "node:assert/strict";
import { columnsFor, masonry } from "./layout.ts";

test("places each item in the shortest column", () => {
  const items = [{ id: 1, aspect: 2 }, { id: 2, aspect: 1 }, { id: 3, aspect: 1 }, { id: 4, aspect: 1 }];
  const { placed } = masonry(items, 2, 100, 10, 0);
  assert.deepEqual(placed.map((p) => p.column), [0, 1, 1, 0]);
  assert.equal(placed[2].top, 110); // under item 2 (100 tall + 10 gap)
});

test("total height is the tallest column", () => {
  const { height } = masonry([{ id: 1, aspect: 1.5 }], 3, 100, 10, 20);
  assert.equal(height, 150 + 20 + 10);
});

test("column count respects minimum width and never drops below 1", () => {
  assert.equal(columnsFor(1000, 236, 16), 4); // (1000+16) / (236+16) = 4.03
  assert.equal(columnsFor(1000, 240, 16), 3);
  assert.equal(columnsFor(100), 1);
});
