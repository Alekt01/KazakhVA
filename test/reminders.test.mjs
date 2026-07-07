import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { parseReminderCommand } from "../lib/reminders.mjs";

function localDate(year, month, day, hour, minute = 0) {
  return new Date(year, month - 1, day, hour, minute, 0, 0);
}

function assertLocalDateParts(value, parts) {
  const date = new Date(value);
  assert.equal(date.getFullYear(), parts.year);
  assert.equal(date.getMonth() + 1, parts.month);
  assert.equal(date.getDate(), parts.day);
  assert.equal(date.getHours(), parts.hour);
  assert.equal(date.getMinutes(), parts.minute);
}

describe("reminder parsing", () => {
  test("parses minute timers", () => {
    const now = localDate(2026, 7, 6, 9, 30);
    const parsed = parseReminderCommand("set timer for 15 minutes", { now });

    assert.equal(parsed.ok, true);
    assert.equal(parsed.type, "timer");
    assert.equal(parsed.text, "Timer for 15 minutes");
    assert.equal(new Date(parsed.dueAt).getTime(), now.getTime() + 15 * 60 * 1000);
  });

  test("parses named timers", () => {
    const now = localDate(2026, 7, 6, 9, 30);
    const parsed = parseReminderCommand("set a 5 minute timer to check tea", { now });

    assert.equal(parsed.ok, true);
    assert.equal(parsed.type, "timer");
    assert.equal(parsed.text, "check tea");
    assert.equal(new Date(parsed.dueAt).getTime(), now.getTime() + 5 * 60 * 1000);
  });

  test("parses tomorrow at 5 as local evening", () => {
    const now = localDate(2026, 7, 6, 9, 30);
    const parsed = parseReminderCommand("remind me tomorrow at 5 to call mom", { now });

    assert.equal(parsed.ok, true);
    assert.equal(parsed.type, "reminder");
    assert.equal(parsed.text, "call mom");
    assertLocalDateParts(parsed.dueAt, {
      year: 2026,
      month: 7,
      day: 7,
      hour: 17,
      minute: 0
    });
  });

  test("parses explicit reminder times", () => {
    const now = localDate(2026, 7, 6, 9, 30);
    const parsedPm = parseReminderCommand("remind me tomorrow at 5 pm to stretch", { now });
    const parsedTwentyFourHour = parseReminderCommand("remind me tomorrow at 17:45 to stretch", { now });

    assert.equal(parsedPm.ok, true);
    assert.equal(parsedTwentyFourHour.ok, true);
    assertLocalDateParts(parsedPm.dueAt, { year: 2026, month: 7, day: 7, hour: 17, minute: 0 });
    assertLocalDateParts(parsedTwentyFourHour.dueAt, { year: 2026, month: 7, day: 7, hour: 17, minute: 45 });
  });

  test("returns a parse error for invalid reminder times", () => {
    const now = localDate(2026, 7, 6, 9, 30);
    const parsed = parseReminderCommand("remind me tomorrow at 25 to call mom", { now });

    assert.equal(parsed.ok, false);
    assert.match(parsed.error, /could not understand/i);
  });

  test("ignores unrelated text", () => {
    assert.equal(parseReminderCommand("what is the weather tomorrow"), null);
  });
});
