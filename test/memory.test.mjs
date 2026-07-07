import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, test } from "node:test";
import { createMemoryStore, deleteMemories, deleteMemory, listMemories, upsertMemory } from "../lib/memory.mjs";

let tempDir;
let memoryStore;

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), "myva-memory-"));
  memoryStore = createMemoryStore({
    dataDir: tempDir,
    storePath: join(tempDir, "store.json")
  });
});

afterEach(async () => {
  await rm(tempDir, { recursive: true, force: true });
});

describe("memory store", () => {
  test("saves long-term memories", async () => {
    const store = await memoryStore.readStore();
    const result = upsertMemory(store, {
      kind: "fact",
      text: "Alexei is building a Kazakh voice assistant.",
      tags: ["project"],
      confidence: 0.95,
      speakerName: "Alexei",
      language: "en",
      source: "test"
    });

    await memoryStore.writeStore(store);

    assert.equal(result.created, true);
    assert.equal(store.memories.length, 1);
    assert.equal(store.memories[0].text, "Alexei is building a Kazakh voice assistant.");
  });

  test("retrieves persisted memories", async () => {
    const store = await memoryStore.readStore();
    upsertMemory(store, {
      kind: "preference",
      text: "Alexei prefers concise assistant responses.",
      tags: ["preference"],
      confidence: 0.9,
      speakerName: "Alexei",
      language: "en",
      source: "test"
    });
    await memoryStore.writeStore(store);

    const reloaded = await memoryStore.readStore();
    const memories = listMemories(reloaded, { speakerName: "Alexei" });

    assert.equal(memories.length, 1);
    assert.equal(memories[0].kind, "preference");
    assert.equal(memories[0].text, "Alexei prefers concise assistant responses.");
  });

  test("deletes persisted memories", async () => {
    const store = await memoryStore.readStore();
    const result = upsertMemory(store, {
      kind: "instruction",
      text: "Always answer language drills with short examples.",
      tags: ["lesson"],
      confidence: 0.85,
      speakerName: "Alexei",
      language: "en",
      source: "test"
    });
    await memoryStore.writeStore(store);

    const reloaded = await memoryStore.readStore();
    const removed = deleteMemory(reloaded, result.memory.id);
    await memoryStore.writeStore(reloaded);

    const afterDelete = await memoryStore.readStore();
    assert.equal(removed?.id, result.memory.id);
    assert.equal(listMemories(afterDelete).length, 0);
  });

  test("deletes matching memories in bulk", async () => {
    const store = await memoryStore.readStore();
    upsertMemory(store, {
      kind: "fact",
      text: "Alexei is practicing Kazakh.",
      tags: ["language"],
      confidence: 0.9,
      speakerName: "Alexei",
      language: "en",
      source: "test"
    });
    upsertMemory(store, {
      kind: "fact",
      text: "Dana is practicing English.",
      tags: ["language"],
      confidence: 0.9,
      speakerName: "Dana",
      language: "en",
      source: "test"
    });

    const removed = deleteMemories(store, (memory) => memory.speakerName === "Alexei");

    assert.equal(removed.length, 1);
    assert.equal(listMemories(store).length, 1);
    assert.equal(listMemories(store)[0].speakerName, "Dana");
  });
});
