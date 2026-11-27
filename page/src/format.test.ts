import { describe, expect, it } from "vitest";
import { proposalOutcome } from "./format";
import type { Proposal } from "./protocol";

const base: Proposal = { current: 0.8, proposed: 0.5, change: -0.3, candidates: 1, check: null, recommended: false, note: null };

describe("the Proposal row's first line", () => {
  it("says what the run kept and why", () => {
    expect(proposalOutcome(base)).toEqual({ kept: "Current kept", reason: "No A/B check: no proposal beat the current score." });
    const won = { ...base, current: 0, proposed: 0.82, change: 0.82, check: { status: "complete", winner: "baseline", reason: null } };
    expect(proposalOutcome(won)).toEqual({ kept: "Current kept", reason: "A/B check finished: the current instructions won." });
    const incomplete = { ...won, check: { status: "incomplete", winner: null, reason: "guest harness version doesn't match the host" } };
    expect(proposalOutcome(incomplete)).toEqual({
      kept: "Current kept for now",
      reason: "A/B check incomplete: guest harness version doesn't match the host.",
    });
    const tie = { ...won, check: { status: "complete", winner: "tie", reason: null } };
    expect(proposalOutcome(tie).reason).toBe("A/B check finished: a tie, so the current instructions stay.");
    const recommended = { ...won, recommended: true, check: { status: "complete", winner: "candidate", reason: null } };
    expect(proposalOutcome(recommended)).toEqual({ kept: "Proposal recommended", reason: "A/B check finished: the proposal won." });
    expect(proposalOutcome({ ...base, proposed: null, change: null })).toEqual({ kept: "Current kept", reason: "No valid proposal." });
  });
});
