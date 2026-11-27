import type { Card, Proposal, ProviderState } from "./protocol";

const TIME = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** "Nov 26, 11:14 AM", in the Mac's time zone. */
export const shortTime = (iso: string) => TIME.format(new Date(iso));

export const score = (value: number) => value.toFixed(2);

/** "+0.20" or "−0.30", with a minus sign rather than a hyphen. */
export const signed = (value: number) => `${value < 0 ? "−" : "+"}${score(Math.abs(value))}`;

/** The second line of a card: run ID, time, turns, and sessions past the first. */
export function meta(card: Card): string {
  const parts = [card.run_id, shortTime(card.created_at)];
  if (card.turns !== null) parts.push(`${card.turns} turns`);
  if (card.session_count !== null && card.session_count > 1) parts.push(`${card.session_count} sessions`);
  return parts.join(" · ");
}

export const ratings = (card: Card) =>
  card.mean_score === null ? null : `${card.rated} rating${card.rated === 1 ? "" : "s"} · mean ${score(card.mean_score)}`;

const sentence = (text: string) => (/[.!?]$/.test(text) ? text : `${text}.`);

/**
 * What the run kept and why, for the first line of the Proposal row. A run recommends a proposal
 * only when the A/B check in the sandbox finished and the proposal won; when the current
 * instructions win, tie, or the check is incomplete, it keeps them, whatever the evaluator
 * predicted, and an incomplete check keeps them only "for now".
 */
export function proposalOutcome(p: Proposal): { kept: string; reason: string } {
  const check = p.check;
  if (p.recommended) {
    return { kept: "Proposal recommended", reason: "A/B check finished: the proposal won." };
  }
  if (p.proposed === null) return { kept: "Current kept", reason: "No valid proposal." };
  if (!check) {
    const beat = p.change !== null && p.change > 0;
    return { kept: "Current kept", reason: beat ? "No A/B check ran." : "No A/B check: no proposal beat the current score." };
  }
  if (check.status !== "complete") {
    const status = check.status.replace(/_/g, " ");
    return { kept: "Current kept for now", reason: sentence(`A/B check ${status}${check.reason ? `: ${check.reason}` : ""}`) };
  }
  if (check.winner === "baseline") return { kept: "Current kept", reason: "A/B check finished: the current instructions won." };
  if (check.winner === "candidate") return { kept: "Current kept", reason: "A/B check finished: the proposal won, but it isn't recommended." };
  return { kept: "Current kept", reason: "A/B check finished: a tie, so the current instructions stay." };
}

export const PROVIDER_WORDS: Record<ProviderState, string> = {
  signed_in: "Signed in",
  signed_out: "Signed out",
  no_key: "No key",
  not_installed: "Not installed",
  unknown: "Couldn't tell",
};
