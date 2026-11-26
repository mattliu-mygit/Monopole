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

/** What the A/B check in the sandbox found, in a few words. */
export function checkWords(check: Proposal["check"]): string {
  if (!check) return "no A/B check";
  if (check.status !== "complete") return `A/B check ${check.status.replace(/_/g, " ")}`;
  if (check.winner === "baseline") return "the current instructions won";
  if (check.winner === "candidate") return "the proposal won";
  return "the A/B check tied";
}

export const PROVIDER_WORDS: Record<ProviderState, string> = {
  signed_in: "Signed in",
  signed_out: "Signed out",
  no_key: "No key",
  not_installed: "Not installed",
  unknown: "Couldn't tell",
};
