import { createRoot, flush } from "solid-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CallError, type PageMessage } from "./protocol";
import { createPageStore, HOLD_MS } from "./store";

/** A store wired to a host the test answers by hand. */
function storeWithHost() {
  const posted: PageMessage[] = [];
  let toPage: (data: unknown) => void = () => {};
  const store = createRoot(() =>
    createPageStore({
      post: (message) => posted.push(message as PageMessage),
      listen: (receive) => {
        toPage = receive;
        return () => {};
      },
    }),
  );
  const send = (message: object) => toPage({ version: 2, ...message });
  /** The id of the last call of action. */
  const lastCall = (action: string) => posted.flatMap((m) => (m.type === "call" && m.action === action ? [m.id] : [])).at(-1)!;
  const held = () => (flush(), store.held());
  return { store, send, lastCall, held };
}

describe("the page store", () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] }));
  afterEach(() => vi.useRealTimers());

  it("holds the action buttons while an action runs, for HOLD_MS after its reply, and HOLD_MS after an open in that time", async () => {
    const { store, send, lastCall, held } = storeWithHost();
    send({ type: "open", route: "/runs/run-1" });
    expect(held()).toBe(false);

    const continued = store.continueRun("run-1");
    expect(held()).toBe(true);
    send({ type: "result", reply_to: lastCall("continue"), json: { format: 1, run_id: "run-1", status: "judging", route: "/runs/run-1" } });
    await continued;
    expect(held()).toBe(true);

    // Prompting Time opens the next item 400 ms after the reply, which holds the buttons 600 ms more.
    vi.advanceTimersByTime(400);
    send({ type: "open", route: "/runs/run-2" });
    vi.advanceTimersByTime(HOLD_MS - 1);
    expect(held()).toBe(true);
    vi.advanceTimersByTime(1);
    expect(held()).toBe(false);

    // An open after the hold ended holds nothing.
    send({ type: "open", route: "/runs/run-3" });
    expect(held()).toBe(false);
  });

  it("rejects a refusal with Monopole's stderr as the reason, and holds after it too", async () => {
    const { store, send, lastCall, held } = storeWithHost();
    const rerun = store.rerun("run-1");
    const stderr = "Can't rerun run-1: judge GPT-5.6 Sol is no longer in the catalog.\n";
    send({ type: "error", reply_to: lastCall("rerun"), message: "bin/monopole exited with 1.", stage: "preview", exit_code: 1, stderr });
    const error = await rerun.catch((e: unknown) => e);
    expect(error).toBeInstanceOf(CallError);
    expect((error as CallError).message).toBe(stderr.trim());
    expect((error as CallError).failure).toEqual({ stage: "preview", exit_code: 1, stderr });
    expect(held()).toBe(true);
    vi.advanceTimersByTime(HOLD_MS);
    expect(held()).toBe(false);
  });

  it("rejects with the host's message when the command wrote nothing on stderr", async () => {
    const { store, send, lastCall } = storeWithHost();
    const rerun = store.rerun("run-1");
    send({ type: "error", reply_to: lastCall("rerun"), message: "Nothing ran: the confirmation was cancelled.", stage: "cancelled" });
    expect(((await rerun.catch((e: unknown) => e)) as CallError).message).toBe("Nothing ran: the confirmation was cancelled.");
  });
});
