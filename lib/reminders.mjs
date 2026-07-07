import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";

export const defaultReminderStore = {
  reminders: []
};

export function freshDefaultReminderStore() {
  return structuredClone(defaultReminderStore);
}

export function normalizeReminderStore(store = {}) {
  return {
    ...freshDefaultReminderStore(),
    ...store,
    reminders: Array.isArray(store.reminders) ? store.reminders : []
  };
}

export function createReminderStore({ dataDir, remindersPath }) {
  async function readReminders() {
    await mkdir(dataDir, { recursive: true });
    try {
      const raw = await readFile(remindersPath, "utf8");
      return normalizeReminderStore(JSON.parse(raw));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const store = freshDefaultReminderStore();
      await writeReminders(store);
      return store;
    }
  }

  async function writeReminders(store) {
    await mkdir(dataDir, { recursive: true });
    await writeFile(remindersPath, JSON.stringify(normalizeReminderStore(store), null, 2), "utf8");
  }

  return { readReminders, writeReminders };
}

export function parseReminderCommand(text, options = {}) {
  const cleaned = String(text || "").trim().replace(/\s+/g, " ");
  if (!cleaned) return null;
  return parseTimerCommand(cleaned, options) || parseDatedReminderCommand(cleaned, options);
}

export function addReminder(store, draft, options = {}) {
  const now = options.now || new Date();
  if (!Array.isArray(store.reminders)) store.reminders = [];
  const reminder = {
    id: randomUUID(),
    type: draft.type || "reminder",
    text: draft.text,
    dueAt: new Date(draft.dueAt).toISOString(),
    status: "pending",
    createdAt: now.toISOString(),
    source: draft.source || "local-command",
    ...(draft.durationMinutes ? { durationMinutes: draft.durationMinutes } : {})
  };
  store.reminders.push(reminder);
  store.reminders.sort((left, right) => new Date(left.dueAt) - new Date(right.dueAt));
  return reminder;
}

export function takeDueReminders(store, options = {}) {
  const now = options.now || new Date();
  const due = [];
  if (!Array.isArray(store.reminders)) store.reminders = [];
  for (const reminder of store.reminders) {
    if (reminder.status !== "pending") continue;
    if (new Date(reminder.dueAt).getTime() > now.getTime()) continue;
    reminder.status = "notified";
    reminder.notifiedAt = now.toISOString();
    due.push(reminder);
  }
  return due;
}

export function formatLocalDateTime(date) {
  const value = new Date(date);
  const pad = (number) => String(number).padStart(2, "0");
  return [
    `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}`,
    `${pad(value.getHours())}:${pad(value.getMinutes())}`
  ].join(" ");
}

function parseTimerCommand(text, options = {}) {
  const now = options.now || new Date();
  const patterns = [
    /^(?:please\s+)?set\s+(?:a\s+)?timer\s+for\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b(?:\s+(?:to|for|called|named)\s+(.+))?$/iu,
    /^(?:please\s+)?set\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\s+timer\b(?:\s+(?:to|for|called|named)\s+(.+))?$/iu
  ];
  const match = patterns.map((pattern) => text.match(pattern)).find(Boolean);
  if (!match) return null;

  const amount = Number(match[1]);
  if (!Number.isFinite(amount) || amount <= 0) {
    return { ok: false, error: "Timer duration must be greater than zero." };
  }

  const unit = match[2].toLocaleLowerCase();
  const durationMinutes = unit.startsWith("hour") || unit.startsWith("hr") ? amount * 60 : amount;
  const dueAt = new Date(now.getTime() + durationMinutes * 60 * 1000);
  const message = cleanReminderText(match[3]) || `Timer for ${formatDurationMinutes(durationMinutes)}`;
  return {
    ok: true,
    type: "timer",
    text: message,
    dueAt: dueAt.toISOString(),
    durationMinutes
  };
}

function parseDatedReminderCommand(text, options = {}) {
  const now = options.now || new Date();
  const directMatch = text.match(/^(?:please\s+)?remind\s+me\s+(today|tomorrow)\s+at\s+(.+?)\s+to\s+(.+)$/iu);
  const reversedMatch = text.match(/^(?:please\s+)?remind\s+me\s+at\s+(.+?)\s+(today|tomorrow)\s+to\s+(.+)$/iu);
  const datePhrase = directMatch?.[1] || reversedMatch?.[2] || "";
  const timeText = directMatch?.[2] || reversedMatch?.[1] || "";
  const message = cleanReminderText(directMatch?.[3] || reversedMatch?.[3] || "");
  if (directMatch || reversedMatch) {
    if (!message) return { ok: false, error: "Reminder text is required." };
    const parsedTime = parseClockTime(timeText);
    if (!parsedTime) return { ok: false, error: `I could not understand the reminder time: ${timeText}` };
    const dayOffset = datePhrase.toLocaleLowerCase() === "tomorrow" ? 1 : 0;
    const dueAt = new Date(now);
    dueAt.setDate(now.getDate() + dayOffset);
    dueAt.setHours(parsedTime.hour, parsedTime.minute, 0, 0);
    if (dueAt.getTime() <= now.getTime()) {
      return { ok: false, error: "That reminder time has already passed." };
    }
    return {
      ok: true,
      type: "reminder",
      text: message,
      dueAt: dueAt.toISOString()
    };
  }
  if (/^(?:please\s+)?remind\s+me\b/iu.test(text)) {
    return { ok: false, error: 'Try: "remind me tomorrow at 5 to call mom".' };
  }
  return null;
}

function parseClockTime(value) {
  const match = String(value || "")
    .trim()
    .match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$/iu);
  if (!match) return null;
  let hour = Number(match[1]);
  const minute = Number(match[2] || 0);
  const meridiem = match[3]?.replace(/\./g, "").toLocaleLowerCase() || "";
  if (!Number.isInteger(hour) || !Number.isInteger(minute) || minute < 0 || minute > 59) return null;

  if (meridiem === "am") {
    if (hour < 1 || hour > 12) return null;
    hour = hour === 12 ? 0 : hour;
  } else if (meridiem === "pm") {
    if (hour < 1 || hour > 12) return null;
    hour = hour === 12 ? 12 : hour + 12;
  } else if (hour >= 1 && hour <= 7) {
    hour += 12;
  }

  if (hour < 0 || hour > 23) return null;
  return { hour, minute };
}

function cleanReminderText(value) {
  return String(value || "")
    .trim()
    .replace(/[.?!]+$/g, "")
    .slice(0, 500);
}

function formatDurationMinutes(minutes) {
  if (Number.isInteger(minutes)) return `${minutes} minute${minutes === 1 ? "" : "s"}`;
  return `${Number(minutes.toFixed(2))} minutes`;
}
