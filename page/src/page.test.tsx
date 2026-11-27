import { fireEvent as dom, screen, waitFor, within } from "@testing-library/dom";
import { flush } from "solid-js";
import { describe, expect, it, vi } from "vitest";
import { renderWithFakeHost } from "./test/connect";

// Solid applies updates in a microtask, so each event is followed by flush() to see its effect.
const fireEvent = {
  click: (element: Element) => (dom.click(element), flush()),
  mouseOver: (element: Element) => (dom.mouseOver(element), flush()),
  keyDown: (element: Element | Document, init: object) => (dom.keyDown(element, init), flush()),
};

const card = (title: string) => screen.findByText(title).then((element) => element.closest("li")!);
const statusButton = async () =>
  within(await card("Session 01a1ad64-7d0")).getByRole("button", { name: /Failed at judging/ });

describe("Monopole's page with the fake host", () => {
  it("lists the runs as cards, read with the runs action after ready, with the last provider check for the sidebar", async () => {
    const host = renderWithFakeHost();
    const review = await card("Lets pause after scoring editing the skill copies");
    expect(host.received[0]).toEqual({ version: 2, type: "ready" });
    expect(host.calls()).toEqual([["runs"], ["providers"]]);
    expect(within(review).getByText("Review needed")).toBeInTheDocument();
    expect(within(review).getByText(/run-review01 · .* · 114 turns/)).toBeInTheDocument();
    expect(within(review).getByText("DeepSeek V4.1 Flash, GPT-5.6 Sol")).toBeInTheDocument();
    expect(within(review).getByText("· 6 ratings · mean 0.71")).toBeInTheDocument();
    expect(within(await card("Fix the flaky login test")).getByText("· Judging 2 of 6 rubrics")).toBeInTheDocument();
  });

  it("shows how the proposal compared with the current instructions", async () => {
    renderWithFakeHost();
    const row = within(await card("Lets pause after scoring editing the skill copies")).getByLabelText("Proposal");
    // Option B: what was kept and why first, the predicted scores on a smaller second line.
    const [outcome, scores] = Array.from(row.children);
    expect(within(outcome as HTMLElement).getByText("Current kept for now")).toBeInTheDocument();
    expect(within(outcome as HTMLElement).getByText("· A/B check incomplete: Sandbox preflight failed.")).toBeInTheDocument();
    expect(scores).toHaveTextContent("current0.15");
    expect(scores).toHaveTextContent("proposed0.35");
    expect(within(scores as HTMLElement).getByText("+0.20")).toBeInTheDocument();
    expect(within(scores as HTMLElement).getByText("predicted by the evaluator · best of 3")).toBeInTheDocument();
    // The evaluator's note shows on the run's own page, not on the card.
    expect(within(row).queryByText("Targets outcome.command and outcome.test.")).not.toBeInTheDocument();
  });

  it("opens a failed run's cause only on a click, and closes it on a click outside or Esc", async () => {
    renderWithFakeHost();
    const button = await statusButton();
    fireEvent.mouseOver(button);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(button).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(button);
    const popover = screen.getByRole("dialog", { name: "Why run-failed01 failed" });
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(within(popover).getByText(/DeepSeek V4.1 Flash failed ValueError/)).toBeInTheDocument();
    expect(within(popover).getByText("Session Outcome Quality")).toBeInTheDocument();
    // A click inside the popover leaves it open.
    fireEvent.click(within(popover).getByText("Rubric or step"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    fireEvent.click(document.body);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    // A second click on the status closes it too.
    fireEvent.click(button);
    fireEvent.click(button);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens the run Prompting Time names, and reads it with run --item", async () => {
    const host = renderWithFakeHost({ openOnReady: "/runs/run-failed01" });
    expect(await screen.findByRole("heading", { name: "Session 01a1ad64-7d0" })).toBeInTheDocument();
    expect(host.calls()).toContainEqual(["run", "--item", "run-failed01"]);
    expect(screen.getByRole("heading", { name: "Stopped at Session Outcome Quality" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /^Judging\s*·\s*Failed$/ })).toBeInTheDocument();
    // The Runs tab stays selected while one run is open.
    expect(screen.getByRole("button", { name: /^Runs/ })).toHaveAttribute("aria-current", "page");
  });

  it("deletes a run from its ⋯ menu through Prompting Time's confirmation and goes back to the list", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm, openOnReady: "/runs/run-failed01" });
    fireEvent.click(await screen.findByRole("button", { name: "More actions for run-failed01" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete run" }));
    await waitFor(() => expect(confirm).toHaveBeenCalledWith("delete", ["--item", "run-failed01"]));
    // The page passes only the item; Prompting Time adds --print and --expect.
    expect(host.calls()).toContainEqual(["delete", "--item", "run-failed01"]);
    await screen.findByText("Lets pause after scoring editing the skill copies");
    expect(screen.queryByText("Session 01a1ad64-7d0")).not.toBeInTheDocument();
  });

  it("shows a cancelled confirmation and keeps the run", async () => {
    renderWithFakeHost({ confirm: () => false, openOnReady: "/runs/run-failed01" });
    fireEvent.click(await screen.findByRole("button", { name: "More actions for run-failed01" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("the confirmation was cancelled");
    expect(screen.getByRole("heading", { name: "Session 01a1ad64-7d0" })).toBeInTheDocument();
  });

  it("can't delete a running run", async () => {
    renderWithFakeHost({ openOnReady: "/runs/run-running1" });
    fireEvent.click(await screen.findByRole("button", { name: "More actions for run-running1" }));
    expect(await screen.findByRole("menuitem", { name: "Delete run" })).toHaveAttribute("aria-disabled", "true");
  });

  it("reads again when Prompting Time says the state folder changed", async () => {
    const host = renderWithFakeHost();
    await card("Fix the flaky login test");
    host.changed(["runs-changed"]);
    await waitFor(() => expect(host.calls()).toEqual([["runs"], ["providers"], ["runs"], ["providers"]]));
  });

  it("checks every provider from the providers page", async () => {
    const host = renderWithFakeHost({ openOnReady: "/providers" });
    expect(await screen.findByText("Not checked yet")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check providers" }));
    expect(await screen.findByText("Codex")).toBeInTheDocument();
    expect(host.calls()).toContainEqual(["check-providers"]);
    const codex = screen.getByText("Codex").closest("li")!;
    expect(within(codex).getByText("Signed out")).toBeInTheDocument();
    expect(within(codex).getByText("Not logged in")).toBeInTheDocument();
    expect(screen.getByText(/^Checked /)).toBeInTheDocument();
  });
});
