import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";

export const defaultStore = {
  profiles: [],
  workflows: [],
  memories: [],
  learning: {
    learners: []
  }
};

export function freshDefaultStore() {
  return structuredClone(defaultStore);
}

export function normalizeStoreShape(store = {}) {
  const defaults = freshDefaultStore();
  const normalized = {
    ...defaults,
    ...store,
    profiles: Array.isArray(store.profiles) ? store.profiles : [],
    workflows: Array.isArray(store.workflows) ? store.workflows : [],
    memories: Array.isArray(store.memories) ? store.memories : [],
    learning: {
      ...defaults.learning,
      ...(store.learning && typeof store.learning === "object" ? store.learning : {})
    }
  };
  if (!Array.isArray(normalized.learning.learners)) normalized.learning.learners = [];
  return normalized;
}

export function createMemoryStore({ dataDir, storePath }) {
  async function readStore() {
    await mkdir(dataDir, { recursive: true });
    try {
      const raw = await readFile(storePath, "utf8");
      return normalizeStoreShape(JSON.parse(raw));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const store = freshDefaultStore();
      await writeStore(store);
      return store;
    }
  }

  async function writeStore(store) {
    await mkdir(dataDir, { recursive: true });
    await writeFile(storePath, JSON.stringify(store, null, 2), "utf8");
  }

  return { readStore, writeStore };
}

export function upsertByName(items, item) {
  const normalized = item.name.toLocaleLowerCase();
  const existingIndex = items.findIndex((entry) => entry.name.toLocaleLowerCase() === normalized);
  if (existingIndex >= 0) {
    items[existingIndex] = { ...items[existingIndex], ...item, updatedAt: new Date().toISOString() };
    return items[existingIndex];
  }
  const next = {
    id: randomUUID(),
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    ...item
  };
  items.push(next);
  return next;
}

export function listMemories(store, filters = {}) {
  const memories = Array.isArray(store.memories) ? store.memories : [];
  return memories.filter((memory) => {
    if (filters.kind && memory.kind !== filters.kind) return false;
    if (filters.speakerName && memory.speakerName !== filters.speakerName) return false;
    if (filters.tag && !memory.tags?.includes(filters.tag)) return false;
    return true;
  });
}

export function deleteMemory(store, id) {
  if (!Array.isArray(store.memories)) store.memories = [];
  const index = store.memories.findIndex((memory) => memory.id === id);
  if (index < 0) return null;
  const [removed] = store.memories.splice(index, 1);
  return removed;
}

export function deleteMemories(store, predicate) {
  if (!Array.isArray(store.memories)) store.memories = [];
  const removed = [];
  store.memories = store.memories.filter((memory) => {
    if (!predicate(memory)) return true;
    removed.push(memory);
    return false;
  });
  return removed;
}

export function normalizeMemoryKind(value) {
  const kind = String(value || "").trim().toLocaleLowerCase();
  if (["preference", "goal", "fact", "project", "language", "person", "instruction"].includes(kind)) return kind;
  return "fact";
}

export function normalizeMemoryTags(tags, normalizeForMatching = defaultNormalizeForMatching) {
  if (!Array.isArray(tags)) return [];
  return [...new Set(tags.map((tag) => normalizeForMatching(tag)).filter(Boolean))]
    .slice(0, 6)
    .map((tag) => tag.slice(0, 32));
}

export function containsSensitiveMemory(text) {
  return /\b(password|passcode|api key|apikey|secret|token|private key|credit card|card number|ssn|social security)\b/i.test(text);
}

export function sameMemoryOwner(left, right, normalizeForMatching = defaultNormalizeForMatching) {
  if (!left?.speakerName || !right?.speakerName) return true;
  return normalizeForMatching(left.speakerName) === normalizeForMatching(right.speakerName);
}

export function upsertMemory(store, candidate, options = {}) {
  if (!Array.isArray(store.memories)) store.memories = [];
  const phraseSimilarity = options.phraseSimilarity || defaultPhraseSimilarity;
  const normalizeForMatching = options.normalizeForMatching || defaultNormalizeForMatching;
  const existing = store.memories.find(
    (memory) => sameMemoryOwner(memory, candidate, normalizeForMatching) && phraseSimilarity(memory.text, candidate.text) >= 0.9
  );
  const now = new Date().toISOString();
  if (existing) {
    existing.kind = candidate.kind || existing.kind;
    existing.text = candidate.text.length > existing.text.length ? candidate.text : existing.text;
    existing.tags = [...new Set([...(existing.tags || []), ...(candidate.tags || [])])].slice(0, 8);
    existing.confidence = Math.max(Number(existing.confidence || 0), candidate.confidence);
    existing.speakerName = existing.speakerName || candidate.speakerName || null;
    existing.language = candidate.language || existing.language || "auto";
    existing.updatedAt = now;
    return { memory: existing, created: false };
  }
  const memory = {
    id: randomUUID(),
    createdAt: now,
    updatedAt: now,
    ...candidate
  };
  store.memories.push(memory);
  if (store.memories.length > 200) store.memories.splice(0, store.memories.length - 200);
  return { memory, created: true };
}

export function summarizeMemory(store, options = {}) {
  const summarizeLearning = options.summarizeLearning || (() => "No language learning profiles yet.");
  const profiles = store.profiles.length
    ? store.profiles.map((profile) => `${profile.name}: ${profile.preferredLanguage || "auto"}; ${profile.notes || "no notes"}`).join("\n")
    : "No people saved yet.";
  const workflows = store.workflows.length
    ? store.workflows.map((workflow) => `${workflow.name}: say "${workflow.trigger}" -> ${workflow.steps.join(" -> ")}`).join("\n")
    : "No workflows saved yet.";
  const memories = store.memories?.length
    ? store.memories
        .slice(-20)
        .map((memory) => {
          const owner = memory.speakerName ? `${memory.speakerName}: ` : "";
          const tags = memory.tags?.length ? ` [${memory.tags.join(", ")}]` : "";
          return `${owner}${memory.text}${tags}`;
        })
        .join("\n")
    : "No long-term memories saved yet.";
  const learning = summarizeLearning(store);
  return `People:\n${profiles}\n\nWorkflows:\n${workflows}\n\nLong-term memories:\n${memories}\n\nLearning:\n${learning}`;
}

function defaultNormalizeForMatching(value) {
  return String(value || "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function defaultTokenOverlapScore(left, right) {
  const leftTokens = new Set(defaultNormalizeForMatching(left).split(" ").filter(Boolean));
  const rightTokens = new Set(defaultNormalizeForMatching(right).split(" ").filter(Boolean));
  if (!leftTokens.size || !rightTokens.size) return 0;
  const overlap = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  return overlap / Math.max(leftTokens.size, rightTokens.size);
}

function defaultLevenshteinDistance(left, right) {
  if (left === right) return 0;
  if (!left) return right.length;
  if (!right) return left.length;

  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  const current = Array.from({ length: right.length + 1 }, () => 0);

  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    current[0] = leftIndex;
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      const substitutionCost = left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1;
      current[rightIndex] = Math.min(
        previous[rightIndex] + 1,
        current[rightIndex - 1] + 1,
        previous[rightIndex - 1] + substitutionCost
      );
    }
    previous.splice(0, previous.length, ...current);
  }

  return previous[right.length];
}

function defaultPhraseSimilarity(left, right) {
  const normalizedLeft = defaultNormalizeForMatching(left);
  const normalizedRight = defaultNormalizeForMatching(right);
  if (!normalizedLeft || !normalizedRight) return 0;
  if (normalizedLeft === normalizedRight) return 1;
  const distance = defaultLevenshteinDistance(normalizedLeft, normalizedRight);
  const characterScore = 1 - distance / Math.max(normalizedLeft.length, normalizedRight.length);
  return Math.max(characterScore, defaultTokenOverlapScore(normalizedLeft, normalizedRight) * 0.95);
}
