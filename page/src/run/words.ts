// Small formatters the run view's stage sections share. Those the web UI also uses come from
// frontend/src/features/runs/runFormat.ts, so the page and the web UI show a run the same way.
export { durationLabel as duration, turnRanges, usageText, words } from "../../../frontend/src/features/runs/runFormat";

/** "1 turn", "3 turns". */
export const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`;

const CLOCK = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });

/** "3:04:05 PM", in the Mac's time zone. */
export const clock = (iso: string) => (Number.isFinite(Date.parse(iso)) ? CLOCK.format(new Date(iso)) : "");

/** "0.75", or "—" for no score. */
export const fixed = (value: number | null | undefined, digits = 2) => (value === null || value === undefined ? "—" : value.toFixed(digits));
