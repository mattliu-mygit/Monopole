import { describe, expect, it } from "vitest";
import { readHostMessage, readPageMessage } from "./protocol";

describe("the page's messages, page protocol 2", () => {
  it("reads a host message only at version 2 with exactly its type's fields", () => {
    const good = [
      { version: 2, type: "result", reply_to: 3, json: { format: 1, runs: [], truncated: false } },
      { version: 2, type: "result", reply_to: 4, json: "Deleted run-1.\n" },
      { version: 2, type: "error", reply_to: 3, message: "There is no run run-9." },
      { version: 2, type: "error", message: "Prompting Time could not run the action." },
      { version: 2, type: "error", reply_to: 5, message: "Nothing ran: the confirmation was cancelled.", stage: "cancelled" },
      { version: 2, type: "error", reply_to: 6, message: "run-1 is Complete", stage: "preview", exit_code: 1, stderr: "run-1 is Complete\n" },
      { version: 2, type: "error", reply_to: 7, message: "half done", stage: "run", exit_code: null, stderr: "" },
      { version: 2, type: "open", route: "/runs/run-1" },
      { version: 2, type: "changed", files: ["runs-changed"] },
      { version: 2, type: "changed", files: null },
    ];
    for (const message of good) expect(readHostMessage(message)).toEqual(message);
    for (const bad of [
      { version: 1, type: "open", route: "/runs/r" },
      { version: 2, type: "status", status: {} },
      { version: 2, type: "result", reply_to: "3", json: null },
      { version: 2, type: "result", json: null },
      { version: 2, type: "error", reply_to: 1 },
      { version: 2, type: "error", message: "m", stage: "confirm" },
      { version: 2, type: "error", message: "m", exit_code: "1" },
      { version: 2, type: "error", message: "m", stderr: null },
      { version: 2, type: "open", route: "runs/r" },
      { version: 2, type: "open", route: "/runs/r", extra: true },
      { version: 2, type: "changed", files: [1] },
      "not an object",
      null,
    ]) {
      expect(readHostMessage(bad)).toBeNull();
    }
  });

  it("reads a page message as Prompting Time does before it runs anything", () => {
    const call = { version: 2, type: "call", id: 1, action: "run", args: ["--item", "run-1"] };
    expect(readPageMessage(call)).toEqual(call);
    expect(readPageMessage({ version: 2, type: "ready" })).toEqual({ version: 2, type: "ready" });
    for (const bad of [
      { ...call, version: 1 },
      { ...call, id: undefined },
      { ...call, id: 1.5 },
      { ...call, action: "" },
      { ...call, args: ["--item", 2] },
      { ...call, run_id: "r" },
      { version: 2, type: "ready", id: 1 },
    ]) {
      expect(readPageMessage(bad)).toBeNull();
    }
  });
});
