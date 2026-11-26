import { Show } from "solid-js";
import { checkWords, score, signed } from "./format";
import type { Proposal } from "./protocol";

/** One score as a short bar on a 0 to 1 track, with its value. */
function Bar(props: { label: string; value: number; proposed?: boolean }) {
  return (
    <>
      <span>{props.label}</span>
      <span class="m-track">
        <i class={{ "m-proposed": !!props.proposed }} style={{ width: `${Math.round(props.value * 100)}%` }} />
      </span>
      <span class="m-mono">{score(props.value)}</span>
    </>
  );
}

/**
 * How the run's best proposal compared with the current instructions: both predicted evaluator
 * scores and the change, the A/B check, whether the run kept the current instructions or
 * recommends the proposal, and the evaluator's note.
 */
export function ProposalRow(props: { proposal: Proposal }) {
  const p = () => props.proposal;
  return (
    <div class="m-proposal" aria-label="Proposal">
      <span class="m-label">Proposal</span>
      <Show
        when={p().current !== null && p().proposed !== null && p().change !== null}
        fallback={<span>No valid proposal</span>}
      >
        <span class="m-bars">
          <Bar label="current" value={p().current!} />
          <Bar label="proposed" value={p().proposed!} proposed />
        </span>
        <strong>{signed(p().change!)}</strong>
        <span class="m-quiet">predicted by the evaluator · best of {p().candidates}</span>
      </Show>
      <span class="m-chip" title={p().check?.reason ?? undefined}>
        {checkWords(p().check)}
      </span>
      <span>{p().recommended ? "Proposal recommended" : "Current kept"}</span>
      <Show when={p().note}>
        {(note) => (
          <span class="m-note" title={note()}>
            · {note()}
          </span>
        )}
      </Show>
    </div>
  );
}
