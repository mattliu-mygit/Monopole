import { fireEvent as dom, screen, waitFor, within } from "@testing-library/dom";
import { flush } from "solid-js";
import { describe, expect, it } from "vitest";
import { renderWithFakeHost } from "./test/connect";

// Solid applies updates in a microtask, so each event is followed by flush() to see its effect.
const fireEvent = {
  click: (element: Element) => (dom.click(element), flush()),
  input: (element: Element, value: string) => (dom.input(element, { target: { value } }), flush()),
};

const tab = (name: RegExp) => screen.getByRole("button", { name });
const titles = () => within(screen.getByRole("list", { name: "Runs" })).getAllByRole("button", { name: /./ }).filter((b) => b.classList.contains("m-title")).map((b) => b.textContent);

describe("Monopole's tab sidebar", () => {
  it("names Monopole with its status, counts runs and signed-out providers, and opens each tab's route", async () => {
    const host = renderWithFakeHost();
    await screen.findByText("Lets pause after scoring editing the skill copies");
    const sidebar = screen.getByRole("navigation", { name: "Monopole" });
    expect(screen.getByText("Running 1 run")).toBeInTheDocument();
    expect(within(sidebar).getByRole("button", { name: /^Runs\s*4$/ })).toHaveAttribute("aria-current", "page");

    fireEvent.click(tab(/^Providers/));
    fireEvent.click(await screen.findByRole("button", { name: "Check providers" }));
    // Codex is signed out; OpenAI's missing key doesn't stop every run, so it isn't counted.
    expect(await within(sidebar).findByRole("button", { name: /^Providers\s*1 signed out$/ })).toHaveAttribute("aria-current", "page");

    fireEvent.click(tab(/^Tools/));
    expect(await screen.findByRole("heading", { name: "Tools" })).toBeInTheDocument();
    fireEvent.click(tab(/^New run/));
    expect(await screen.findByRole("heading", { name: "New run" })).toBeInTheDocument();
    expect(host.calls()).toContainEqual(["catalog"]);
  });

  it("selects the tab Prompting Time opens", async () => {
    const host = renderWithFakeHost({ openOnReady: "/trends" });
    expect(await screen.findByRole("heading", { name: "Trends" })).toBeInTheDocument();
    expect(tab(/^Trends/)).toHaveAttribute("aria-current", "page");
    host.open("/providers");
    await waitFor(() => expect(tab(/^Providers/)).toHaveAttribute("aria-current", "page"));
    expect(screen.getByRole("heading", { name: "Providers" })).toBeInTheDocument();
  });
});

describe("the run list's search and filters", () => {
  it("searches titles, run IDs, and judges, and filters by status group with counts", async () => {
    renderWithFakeHost();
    await screen.findByText("Lets pause after scoring editing the skill copies");
    const chips = screen.getByRole("group", { name: "Filter by status" });
    expect(within(chips).getAllByRole("button").map((chip) => chip.textContent)).toEqual([
      "All · 4",
      "Failed · 1",
      "Waiting for you · 1",
      "Review needed · 1",
      "Running · 1",
      "Done · 0",
    ]);

    fireEvent.click(within(chips).getByRole("button", { name: "Failed · 1" }));
    expect(titles()).toEqual(["Session 01a1ad64-7d0"]);
    fireEvent.click(within(chips).getByRole("button", { name: "Failed · 1" }));
    expect(titles()).toHaveLength(4);

    const search = screen.getByRole("searchbox", { name: "Search runs" });
    fireEvent.input(search, "run-wait");
    expect(titles()).toEqual(["Speed up the session list"]);
    // Every card has the same two judges, so a judge's name keeps them all.
    fireEvent.input(search, "gpt-5.6");
    expect(titles()).toHaveLength(4);
    fireEvent.input(search, "flaky");
    expect(within(chips).getByRole("button", { name: "Running · 1" })).toBeInTheDocument();
    expect(within(chips).getByRole("button", { name: "All · 1" })).toBeInTheDocument();

    fireEvent.click(within(chips).getByRole("button", { name: "Done · 0" }));
    expect(screen.getByText(/No runs match/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear the search and filter" }));
    expect(titles()).toHaveLength(4);
  });

  it("keeps the search and filter while a run is open", async () => {
    renderWithFakeHost();
    await screen.findByText("Lets pause after scoring editing the skill copies");
    fireEvent.input(screen.getByRole("searchbox", { name: "Search runs" }), "flaky");
    fireEvent.click(screen.getByRole("button", { name: "Fix the flaky login test" }));
    await screen.findByRole("heading", { name: "Fix the flaky login test" });
    fireEvent.click(tab(/^Runs/));
    expect(await screen.findByRole("searchbox", { name: "Search runs" })).toHaveValue("flaky");
    expect(titles()).toEqual(["Fix the flaky login test"]);
  });
});

describe("a run's stages", () => {
  it("shows a failed run's setup, scoring, judging activity, failure cause, windows, and reviews", async () => {
    renderWithFakeHost({ openOnReady: "/runs/run-failed01" });
    await screen.findByRole("heading", { name: "Session 01a1ad64-7d0" });

    const setup = screen.getByRole("region", { name: "Sessions and settings" });
    expect(within(setup).getByText("351 turns across 1 session")).toBeInTheDocument();
    expect(within(setup).getByRole("table", { name: "Models" })).toHaveTextContent("Judge 2GPT-5.6 Sol");

    const scoring = screen.getByRole("region", { name: /^Scoring/ });
    expect(within(scoring).getByText("Scored turns · 1")).toBeInTheDocument();
    expect(within(scoring).getByText("outcome.tests_passed")).toBeInTheDocument();

    const judging = screen.getByRole("region", { name: /^Judging/ });
    const activity = within(judging).getByRole("list", { name: "Judging activity" });
    // The first event is a debug event, hidden until asked for.
    expect(within(activity).queryByText("Judging started")).not.toBeInTheDocument();
    expect(within(activity).getByText("wandb:DeepSeek-V4.1-Flash · attempt 2 of 3 · provider 429")).toBeInTheDocument();
    fireEvent.click(within(judging).getByRole("button", { name: "Show debug events" }));
    expect(within(activity).getByText("Judging started")).toBeInTheDocument();

    expect(within(judging).getByRole("progressbar", { name: "Rubrics reviewed" })).toHaveAttribute("aria-valuenow", "1");
    expect(within(judging).getByText("Failed reviews · 1")).toBeInTheDocument();
    expect(within(judging).getByText('Cause: wandb:DeepSeek-V4.1-Flash · ValueError: unknown evidence IDs: ["e224"]')).toBeInTheDocument();
    expect(within(judging).getByText("Window 2").closest("li")).toHaveTextContent("Window 2 · turns 4–5, plus turn 3 for context · 27,000 estimated tokens");
    expect(within(judging).getByText("Rubric reviews · 2")).toBeInTheDocument();
    expect(within(judging).getByText("Ran the tests after each change.")).toBeInTheDocument();

    expect(screen.getByRole("region", { name: /^Reflection/ })).toHaveTextContent("Reflection hasn't started.");
  });

  it("shows reflection's proposals, evaluator reasons, A/B check, and rejected attempts", async () => {
    renderWithFakeHost({ openOnReady: "/runs/run-review01" });
    await screen.findByRole("heading", { name: "Lets pause after scoring editing the skill copies" });
    const reflection = screen.getByRole("region", { name: /^Reflection/ });
    expect(within(reflection).getByText("Proposal attempt 2 was rejected")).toBeInTheDocument();
    const proposals = within(reflection).getByRole("table");
    expect(within(proposals).getAllByRole("row").map((row) => row.textContent)).toEqual([
      "ProposalWriterPredictedChange",
      "Current instructions—0.150",
      "Proposal 1wandb:GLM-5.30.350+0.20best, not confirmed",
      "Proposal 2wandb:GLM-5.30.200+0.05",
    ]);
    expect(within(reflection).getByText("A/B check incomplete")).toBeInTheDocument();
    expect(within(reflection).getByText("Sandbox preflight failed")).toBeInTheDocument();
    expect(within(reflection).getByText("Rejected proposal attempts · 1")).toBeInTheDocument();
    expect(within(reflection).getByText("Proposal 1 · predicted 0.350 by wandb:DeepSeek-V4.1-Flash")).toBeInTheDocument();
  });

  it("keeps a fold the user opened open when the run is read again", async () => {
    const host = renderWithFakeHost({ openOnReady: "/runs/run-failed01" });
    await screen.findByRole("heading", { name: "Session 01a1ad64-7d0" });
    const fold = screen.getByText("Scored turns · 1").parentElement!.querySelector("details")!;
    fold.open = true;
    const reads = host.calls().filter(([action]) => action === "run").length;
    host.changed(["runs-changed"]);
    await waitFor(() => expect(host.calls().filter(([action]) => action === "run").length).toBe(reads + 1));
    await new Promise((resolve) => setTimeout(resolve, 0));
    flush();
    expect(screen.getByText("Scored turns · 1").parentElement!.querySelector("details")).toBe(fold);
    expect(fold.open).toBe(true);
  });
});

describe("Trends", () => {
  it("reads the analysis once when it opens, and again only on Read again", async () => {
    const host = renderWithFakeHost({ openOnReady: "/trends" });
    const changes = await screen.findByRole("list", { name: "Recent changes" });
    expect(within(changes).getByRole("listitem")).toHaveTextContent("outcome.tests_passedRegression0.90 → 0.70−0.20significant · n=40");

    const scores = screen.getByRole("region", { name: "Scores · 3" });
    const row = within(scores).getByRole("row", { name: /^outcome.tests_passed/ });
    expect(row).toHaveTextContent("outcome.tests_passed0.8080.0%0.70–0.9040−0.20");
    expect(within(scores).getByRole("row", { name: /^judge.verification/ })).toHaveTextContent("4 · too few to trust");

    const configs = screen.getByRole("region", { name: "By instruction configuration · 2" });
    expect(within(configs).getByRole("columnheader", { name: /^a1b2c3d4/ })).toBeInTheDocument();
    expect(within(configs).getByText("0.90")).toHaveClass("m-best");
    expect(screen.getByText("Tests passed less often this week. Run the suite before ending a turn.")).toBeInTheDocument();

    const trendReads = () => host.calls().filter(([action]) => action === "trends").length;
    expect(trendReads()).toBe(1);
    host.changed(["runs-changed"]);
    fireEvent.click(tab(/^Runs/));
    fireEvent.click(tab(/^Trends/));
    await screen.findByRole("list", { name: "Recent changes" });
    expect(trendReads()).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: "Read again" }));
    await waitFor(() => expect(trendReads()).toBe(2));
  });
});
