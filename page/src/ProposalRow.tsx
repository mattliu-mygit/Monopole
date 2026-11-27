import { Show } from "solid-js";
import { proposalOutcome, score, signed } from "./format";
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
 * How the run's best proposal compared with the current instructions. The first line says what
 * the run kept and why, from the A/B check; the second, smaller line has the evaluator's predicted
 * scores for the current instructions and the best proposal, and the change. With `note`, the
 * evaluator's one-line note follows, as on the run's own page.
 */
export function ProposalRow(props: { proposal: Proposal; note?: boolean }) {
  const p = () => props.proposal;
  const outcome = () => proposalOutcome(p());
  return (
    <div class="m-proposal" aria-label="Proposal">
      <div class="m-proposal-outcome">
        <span class="m-label">Proposal</span>
        <strong>{outcome().kept}</strong>
        <span>· {outcome().reason}</span>
      </div>
      <Show when={p().current !== null && p().proposed !== null && p().change !== null}>
        <div class="m-proposal-scores">
          <span class="m-bars">
            <Bar label="current" value={p().current!} />
            <Bar label="proposed" value={p().proposed!} proposed />
          </span>
          <span class="m-mono">{signed(p().change!)}</span>
          <span>predicted by the evaluator · best of {p().candidates}</span>
        </div>
      </Show>
      <Show when={props.note && p().note}>{(note) => <p class="m-quiet">{note()}</p>}</Show>
    </div>
  );
}
