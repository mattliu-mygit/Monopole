import { fireEvent as dom, screen, waitFor, within } from "@testing-library/dom";
import { flush } from "solid-js";
import { describe, expect, it, vi } from "vitest";
import { renderWithFakeHost } from "./test/connect";

// Checking runs in the list and acting on all of them at once.

const click = (element: Element, init: object = {}) => (dom.click(element, init), flush());
const key = (k: string, init: object = {}, target: Element = document.activeElement ?? document.body) => (dom.keyDown(target, { key: k, ...init }), flush());
const boxes = () => screen.queryAllByRole("checkbox") as HTMLInputElement[];
const box = (runId: string) => screen.getByRole("checkbox", { name: `Check ${runId}` }) as HTMLInputElement;
const bar = () => screen.queryByRole("toolbar", { name: "Checked runs" });
const barButton = (name: string) => within(bar()!).queryByRole("button", { name });
const ready = () => screen.findByText("Lets pause after scoring editing the skill copies");

/** Selects the run with a click on its card, then checks it with x. */
function check(runId: string) {
  click(document.querySelector(`[data-run="${runId}"] .m-meta`)!);
  key("x", {}, document.body);
}

describe("checking runs", () => {
  it("shows the selected card's box, every box once one is checked, and clears them with Esc", async () => {
    renderWithFakeHost();
    await ready();
    expect(boxes()).toHaveLength(0);

    key("j", {}, document.body);
    expect(boxes()).toHaveLength(1);
    key("x");
    expect(box("run-review01").checked).toBe(true);
    expect(boxes()).toHaveLength(4);
    expect(within(bar()!).getByText("1 selected")).toBeInTheDocument();
    key("x");
    expect(bar()).toBeNull();

    key("x");
    key("Escape");
    expect(bar()).toBeNull();
    expect(boxes()).toHaveLength(1);
  });

  it("closes an open menu with Esc before it clears the checks", async () => {
    renderWithFakeHost();
    await ready();
    check("run-review01");
    click(screen.getByRole("button", { name: "More actions for run-failed01" }));
    key("Escape", {}, await screen.findByRole("menu"));
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull());
    expect(box("run-review01").checked).toBe(true);
    key("Escape", {}, document.body);
    expect(bar()).toBeNull();
  });

  it("selects a card with a click outside its controls, and checks a range with Shift+click", async () => {
    renderWithFakeHost();
    await ready();
    check("run-review01");
    click(box("run-running1"), { shiftKey: true });
    expect(boxes().map((input) => input.checked)).toEqual([true, true, true, false]);
    expect(within(bar()!).getByText("3 selected")).toBeInTheDocument();
    // A plain click toggles one box.
    click(box("run-failed01"));
    expect(boxes().map((input) => input.checked)).toEqual([true, false, true, false]);
    // The list's keys still work from a focused box.
    box("run-failed01").focus();
    key("Escape");
    expect(bar()).toBeNull();
  });

  it("checks every run the search and filter show with ⌘A, and a new filter clears them", async () => {
    renderWithFakeHost();
    await ready();
    click(screen.getByRole("button", { name: "Waiting for you · 1" }));
    key("a", { metaKey: true }, document.body);
    expect(within(bar()!).getByText("1 selected")).toBeInTheDocument();
    expect(box("run-waiting1").checked).toBe(true);
    // In the search, ⌘A selects its text instead.
    key("a", { metaKey: true }, screen.getByRole("searchbox", { name: "Search runs" }));
    click(screen.getByRole("button", { name: "All · 4" }));
    expect(bar()).toBeNull();
    key("a", { ctrlKey: true }, document.body);
    expect(within(bar()!).getByText("4 selected")).toBeInTheDocument();
  });

  it("offers only the actions every checked run allows", async () => {
    renderWithFakeHost();
    await ready();
    check("run-waiting1");
    expect(barButton("Continue")).toBeInTheDocument();
    expect(barButton("Delete")).toBeNull();
    expect(barButton("Compare")).toBeNull();
    check("run-review01");
    expect(barButton("Continue")).toBeNull();
    expect(barButton("Compare")).toBeInTheDocument();
    expect(barButton("Rerun")).toBeInTheDocument();
  });
});

describe("acting on checked runs", () => {
  it("deletes the checked runs after one confirmation and says so", async () => {
    const confirm = vi.fn(() => true);
    renderWithFakeHost({ confirm });
    await ready();
    check("run-review01");
    check("run-failed01");
    click(barButton("Delete")!);

    expect(await screen.findByText("Deleted 2 runs.")).toBeInTheDocument();
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(confirm).toHaveBeenCalledWith("delete", ["--item", "run-review01", "--item", "run-failed01"]);
    await waitFor(() => expect(document.querySelector('[data-run="run-review01"]')).toBeNull());
    expect(bar()).toBeNull();
  });

  it("reruns the checked runs, keeps checked the one the server refused, and names why", async () => {
    const confirm = vi.fn(() => true);
    const host = renderWithFakeHost({ confirm, serverRefuses: { "run-failed01": "judge GPT-5.6 Sol is no longer in the catalog" } });
    await ready();
    check("run-review01");
    check("run-failed01");
    click(barButton("Rerun")!);

    expect(await screen.findByText("Started 1 rerun. 1 failed: run-failed01: judge GPT-5.6 Sol is no longer in the catalog")).toBeInTheDocument();
    expect(host.calls()).toContainEqual(["rerun", "--item", "run-review01", "--item", "run-failed01"]);
    expect(within(bar()!).getByText("1 selected")).toBeInTheDocument();
    expect(box("run-failed01").checked).toBe(true);
    // The list stays open; it doesn't jump to a new run.
    expect(host.calls().some((call) => call[0] === "run")).toBe(false);
  });

  it("keeps the checks and shows why when the confirmation is cancelled", async () => {
    renderWithFakeHost({ confirm: () => false });
    await ready();
    check("run-waiting1");
    click(barButton("Continue")!);

    expect(await screen.findByRole("alert")).toHaveTextContent("Nothing ran: the confirmation was cancelled.");
    expect(box("run-waiting1").checked).toBe(true);
  });

  it("compares two checked runs' rubric ratings and proposal scores", async () => {
    const host = renderWithFakeHost();
    await ready();
    check("run-review01");
    check("run-failed01");
    click(barButton("Compare")!);

    const table = await screen.findByRole("table");
    expect(host.calls()).toEqual(expect.arrayContaining([["run", "--item", "run-review01"], ["run", "--item", "run-failed01"]]));
    const row = (name: string) => within(table).getByRole("row", { name: new RegExp(`^${name}`) }).textContent;
    expect(row("Verification Discipline")).toBe("Verification Discipline0.750.75+0.00");
    expect(row("Mean rating")).toBe("Mean rating0.71——");
    expect(row("Proposal: current instructions")).toBe("Proposal: current instructions0.15——");
    expect(row("Proposal: proposed")).toBe("Proposal: proposed0.35——");

    click(screen.getByRole("button", { name: "More Compare runs actions" }));
    click(await screen.findByRole("menuitem", { name: "Back to runs" }));
    // The checks stay for the next action.
    expect(await screen.findByText("2 selected")).toBeInTheDocument();
  });
});
