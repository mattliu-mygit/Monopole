import { fireEvent as dom, screen, waitFor, within } from "@testing-library/dom";
import { flush } from "solid-js";
import { describe, expect, it, vi } from "vitest";
import { CODEX_WARNING } from "./Providers";
import { renderWithFakeHost } from "./test/connect";

// Solid applies updates in a microtask, so each event is followed by flush() to see its effect.
const fireEvent = {
  click: (element: Element) => (dom.click(element), flush()),
  input: (element: Element, value: string) => (dom.input(element, { target: { value } }), flush()),
  change: (element: Element, value: string) => (dom.change(element, { target: { value } }), flush()),
};

// After an action, the page holds the action buttons disabled for HOLD_MS, so a test waits for the
// next button it presses to be enabled.
const enabled = (find: () => HTMLElement) =>
  waitFor(
    () => {
      const button = find();
      expect(button).toBeEnabled();
      return button;
    },
    { timeout: 2_000 },
  );

const select = (label: string) => screen.getByLabelText(label) as HTMLSelectElement;
const checkbox = (name: string | RegExp) => screen.getByRole("checkbox", { name }) as HTMLInputElement;

describe("rerunning a run", () => {
  it("reruns a card's run through Prompting Time's confirmation and opens the new run", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm });
    fireEvent.click(await screen.findByRole("button", { name: "Rerun run-review01" }));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("rerun", ["--item", "run-review01"]));
    await waitFor(() => expect(host.calls()).toContainEqual(["run", "--item", "run-new4"]));
    expect(await screen.findByRole("button", { name: "Edit and rerun" })).toBeInTheDocument();
  });

  it("opens Edit and rerun from a card's ⋯ menu", async () => {
    const host = renderWithFakeHost();
    fireEvent.click(await screen.findByRole("button", { name: "More actions for run-failed01" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Edit and rerun" }));
    expect(await screen.findByText("from run-failed01's settings")).toBeInTheDocument();
    expect(host.calls()).toContainEqual(["run", "--item", "run-failed01"]);
  });

  it("fills New run with the old run's settings for Edit and rerun, and starts what it shows", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm, openOnReady: "/runs/run-failed01" });
    fireEvent.click(await screen.findByRole("button", { name: "Edit and rerun" }));

    expect(await screen.findByText("from run-failed01's settings")).toBeInTheDocument();
    expect(host.calls()).toEqual(expect.arrayContaining([["catalog"], ["sessions", "--limit", "50"]]));
    expect(select("Judge 1").value).toBe("wandb:DeepSeek-V4.1-Flash");
    expect(select("Judge 2").value).toBe("codex:gpt-5.6-sol");
    expect(select("Judge 3").value).toBe("");
    expect(select("Proposal writer").value).toBe("wandb:GLM-5.3");
    expect(select("Evaluator").value).toBe("wandb:DeepSeek-V4.1-Flash");
    expect(checkbox(/^Session 01a1ad64-7d0/).checked).toBe(true);
    expect(checkbox(/^Review this stack of PRs/).checked).toBe(false);
    expect(checkbox("Continue automatically (judging calls the judge models)").checked).toBe(true);

    fireEvent.change(select("Judge 3"), "wandb:Kimi-K2.6");
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));

    await waitFor(() => expect(host.setups).toHaveLength(1));
    expect(host.setups[0]).toEqual({
      selection: { session_ids: ["01a1ad64-7d00-75f7-b10f-e2e0471fdf58"], since: null, until: null, timezone: null },
      config: {
        model_catalog_version: "models-2",
        rubric_catalog_version: "rubrics-2",
        judge_models: ["wandb:DeepSeek-V4.1-Flash", "codex:gpt-5.6-sol", "wandb:Kimi-K2.6"],
        challenge_judge_models: ["wandb:DeepSeek-V4.1-Flash"],
        proposal_model: "wandb:GLM-5.3",
        proposal_evaluator_model: "wandb:DeepSeek-V4.1-Flash",
        rubrics: [],
        candidate_budget: 3,
        force: false,
      },
      auto_run: true,
    });
    expect(confirm).toHaveBeenCalledWith("create-run", [JSON.stringify(host.setups[0])]);
    await waitFor(() => expect(host.calls()).toContainEqual(["run", "--item", "run-new4"]));
  });

  it("starts a blank New run on the catalog's recommendations with no session chosen", async () => {
    renderWithFakeHost({ openOnReady: "/new" });
    expect(await screen.findByRole("heading", { name: "New run" })).toBeInTheDocument();
    expect(select("Judge 1").value).toBe("wandb:Kimi-K2.6");
    expect(select("Judge 2").value).toBe("wandb:GLM-5.3");
    expect(select("Proposal writer").value).toBe("wandb:Kimi-K2.6");
    expect(select("Evaluator").value).toBe("wandb:GLM-5.3");
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled();
    fireEvent.click(checkbox(/^Review this stack of PRs/));
    expect(screen.getByRole("button", { name: "Start run" })).toBeEnabled();
  });
});

describe("continuing a waiting run", () => {
  it("continues a card's waiting run through Prompting Time's confirmation and opens it", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm });
    // Only a run that waits between stages offers Continue.
    expect(await screen.findByRole("button", { name: "Continue run-waiting1" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue run-running1" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Continue run-waiting1" }));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("continue", ["--item", "run-waiting1"]));
    await waitFor(() => expect(host.calls()).toContainEqual(["run", "--item", "run-waiting1"]));
    expect(await screen.findByRole("heading", { name: "Speed up the session list" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
  });

  it("offers Continue on the run page and holds the action buttons after it", async () => {
    renderWithFakeHost({ confirm: () => false, openOnReady: "/runs/run-waiting1" });
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Nothing ran: the confirmation was cancelled.");
    // The reply came back, and the buttons still wait, so a second click can't land on another item.
    expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Rerun" })).toBeDisabled();
    await enabled(() => screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByRole("button", { name: "Rerun" })).toBeEnabled();
  });
});

describe("provider sign-in", () => {
  it("warns before Codex's sign-in starts, and starts it only from the warning", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm, openOnReady: "/providers", holdSignIn: true });
    fireEvent.click(await screen.findByRole("button", { name: "Check all" }));
    const codex = (await screen.findByText("Codex")).closest("li")!;
    fireEvent.click(await enabled(() => within(codex).getByRole("button", { name: "Sign in" })));

    const warning = screen.getByRole("dialog", { name: "Sign in to Codex" });
    expect(within(warning).getByText(CODEX_WARNING)).toBeInTheDocument();
    expect(CODEX_WARNING).toBe("Starting sign-in removes the current Codex login, even if you cancel.");
    expect(host.calls().some(([action]) => action === "sign-in")).toBe(false);

    fireEvent.click(await enabled(() => within(warning).getByRole("button", { name: "Start sign-in" })));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("sign-in", ["--item", "codex"]));

    // While the browser sign-in waits: the fallback link and a field for a pasted code. The other
    // buttons stay usable; only Sign in waits.
    const panel = await screen.findByRole("dialog", { name: "Signing in with codex login" });
    expect(screen.getByRole("button", { name: "Check all" })).toBeEnabled();
    expect(within(codex).getByRole("button", { name: "Sign in" })).toBeDisabled();
    expect(await within(panel).findByText("https://codex.example/sign-in")).toBeInTheDocument();
    fireEvent.input(within(panel).getByLabelText("Code from the browser, if it shows one"), "abc123");
    fireEvent.click(within(panel).getByRole("button", { name: "Submit code" }));
    await waitFor(() => expect(host.codes).toEqual(["abc123"]));

    host.finishSignIn();
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(within(screen.getByText("Codex").closest("li")!).getByText("Signed in")).toBeInTheDocument();
  });

  it("checks and saves a key and never shows it", async () => {
    const host = renderWithFakeHost({ openOnReady: "/providers" });
    fireEvent.click(await screen.findByRole("button", { name: "Check all" }));
    const openai = (await screen.findByText("OpenAI")).closest("li")!;
    fireEvent.click(within(openai).getByRole("button", { name: "Add key" }));
    const field = screen.getByLabelText("API key") as HTMLInputElement;
    expect(field.type).toBe("password");

    fireEvent.input(field, "bad-key");
    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Check and save" })));
    expect(await screen.findByText(/refused the key \(HTTP 401\); nothing saved/)).toBeInTheDocument();

    fireEvent.input(field, "sk-good");
    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Check and save" })));
    expect(await screen.findByText("Saved in Monopole's state folder.")).toBeInTheDocument();
    // Only stage-key, which exits at once, gets the key; set-key, which checks it, doesn't.
    expect(host.calls().slice(-2)).toEqual([
      ["stage-key", "--item", "openai", "--key", "sk-good"],
      ["set-key", "--item", "openai"],
    ]);
    expect(document.body.innerHTML).not.toContain("sk-good");
    expect(within(screen.getByText("OpenAI").closest("li")!).getByText("Signed in")).toBeInTheDocument();
  });
});

describe("Tools", () => {
  it("runs score and backfill through the confirmation and reads an insight table without one", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm, openOnReady: "/tools" });
    expect(await screen.findByRole("heading", { name: "Tools" })).toBeInTheDocument();
    await waitFor(() => expect(host.calls()).toContainEqual(["tools"]));

    fireEvent.change(select("Since"), "168");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("score", ["--hours", "168"]));
    expect(await screen.findByText("Scored 3 turns, 1 sessions, 0 subagents (12 scores). Wrote 12.")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Backfill" })).toBeDisabled();
    fireEvent.input(screen.getByLabelText("Start"), "2026-10-01");
    fireEvent.input(screen.getByLabelText("End"), "2026-10-31");
    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Backfill" })));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("backfill", ["--start", "2026-10-01", "--end", "2026-10-31"]));

    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Check now" })));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("monitor", []));

    // While a tool runs, here waiting for its confirmation, only its own button waits.
    let confirmMonitor: (yes: boolean) => void = () => {};
    confirm.mockImplementationOnce((() => new Promise<boolean>((resolve) => (confirmMonitor = resolve))) as never);
    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Check now" })));
    expect(await screen.findByRole("button", { name: "Running…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Score" })).toBeEnabled();
    confirmMonitor(true);
    await enabled(() => screen.getByRole("button", { name: "Check now" }));

    fireEvent.click(screen.getByRole("tab", { name: "Judge agreement" }));
    fireEvent.click(await enabled(() => screen.getByRole("button", { name: "Read" })));
    expect(await screen.findByText("Table for judges")).toBeInTheDocument();
    expect(confirm).not.toHaveBeenCalledWith("insights", expect.anything());
  });
});
