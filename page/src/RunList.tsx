import { createSignal, For, onSettled, Show } from "solid-js";
import { Button, EllipsisIcon, Menu, StatusDot } from "@prompting-time/ui";
import { meta, ratings } from "./format";
import type { Card } from "./protocol";
import { ProposalRow } from "./ProposalRow";
import type { PageStore } from "./store";

/** What a run card and the run page can do with a run: Continue a run that waits between stages,
 * Rerun with the same settings, open New run filled in with them, and delete; Continue, Rerun, and
 * Delete confirm in Prompting Time first, and `held` says when their buttons wait. */
export function runActions(store: PageStore, setError: (error: string | null) => void) {
  const attempt = (work: () => Promise<unknown>) => async () => {
    setError(null);
    try {
      await work();
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };
  return {
    held: store.held,
    continueRun: (runId: string) => attempt(() => store.continueRun(runId))(),
    rerun: (runId: string) => attempt(() => store.rerun(runId))(),
    editAndRerun: (runId: string) => store.open(`/new?from=${encodeURIComponent(runId)}`),
    remove: (runId: string) => attempt(() => store.deleteRun(runId))(),
  };
}

/**
 * The runs, newest first, as tight cards in one column: the session's first prompt, the run ID,
 * time, and turns, the judges and mean rating, and for a run that reached reflection, a Proposal
 * row. A failed run's status is a button that opens its cause in a popover; only a click opens it,
 * and a click elsewhere or Esc closes it. Each card has Rerun, Continue when the run waits between
 * stages, and Edit and rerun and Delete in its ⋯ menu.
 */
export function RunList(props: { store: PageStore }) {
  const [openRun, setOpenRun] = createSignal<string | null>(null);
  const [actionError, setActionError] = createSignal<string | null>(null);
  const actions = runActions(props.store, setActionError);

  onSettled(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (!target?.closest(".m-popover, .m-status-button")) setOpenRun(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpenRun(null);
    };
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  });

  return (
    <Show
      when={props.store.state.runs}
      fallback={<p class="m-hint" role="status">Reading runs…</p>}
    >
      {(list) => (
        <Show when={list().runs.length > 0} fallback={<p class="m-hint">No runs yet. Start one from New run.</p>}>
          <Show when={actionError()}>
            {(error) => (
              <p class="m-error" role="alert">
                {error()}
              </p>
            )}
          </Show>
          <ul class="m-runs" aria-label="Runs">
            <For each={list().runs}>
              {(card) => (
                <RunCard
                  card={card}
                  open={openRun() === card.run_id}
                  onToggle={() => setOpenRun(openRun() === card.run_id ? null : card.run_id)}
                  onOpenRun={() => props.store.open(`/runs/${encodeURIComponent(card.run_id)}`)}
                  actions={actions}
                />
              )}
            </For>
          </ul>
          <Show when={list().truncated}>
            <p class="m-hint">Older runs are left out.</p>
          </Show>
        </Show>
      )}
    </Show>
  );
}

function RunCard(props: {
  card: Card;
  open: boolean;
  onToggle: () => void;
  onOpenRun: () => void;
  actions: ReturnType<typeof runActions>;
}) {
  const card = () => props.card;
  const menu = () => [
    { value: "edit", label: "Edit and rerun" },
    { value: "delete", label: "Delete", disabled: !card().deletable || props.actions.held() },
  ];
  const popoverId = () => `cause-${card().run_id}`;
  return (
    <li class="m-card" data-group={card().group ?? "draft"}>
      <div class="m-card-body">
        <button type="button" class="m-title" onClick={props.onOpenRun}>
          {card().title}
        </button>
        <div class="m-meta m-mono">{meta(card())}</div>
        <div class="m-facts">
          <Show when={card().judges.length > 0}>
            <span>{card().judges.map((judge) => judge.label).join(", ")}</span>
          </Show>
          <Show when={ratings(card())}>{(text) => <strong>· {text()}</strong>}</Show>
          <Show when={card().failure}>{(failure) => <span class="m-failed">· stopped at {failure().stopped_at}</span>}</Show>
          <Show when={card().progress}>{(progress) => <span>· {progress()}</span>}</Show>
        </div>
        <Show when={card().proposal}>{(proposal) => <ProposalRow proposal={proposal()} />}</Show>
      </div>
      <div class="m-card-side">
        <Show
          when={card().failure}
          fallback={
            <span class="m-status">
              <StatusDot status={card().dot} />
              {card().label}
            </span>
          }
        >
          {(failure) => (
            <>
              <button
                type="button"
                class="m-status m-status-button"
                aria-expanded={props.open ? "true" : "false"}
                aria-controls={popoverId()}
                onClick={props.onToggle}
              >
                <StatusDot status={card().dot} />
                {card().label}
                <span class="m-caret" aria-hidden="true">▾</span>
              </button>
              <Show when={props.open}>
                <div class="m-popover" id={popoverId()} role="dialog" aria-label={`Why ${card().run_id} failed`}>
                  <h3 class="m-failed">{card().label}</h3>
                  <p>{failure().cause}</p>
                  <dl class="m-kv">
                    <dt>Rubric or step</dt>
                    <dd>{failure().stopped_at}</dd>
                    <dt>Judges</dt>
                    <dd>{card().judges.map((judge) => judge.label).join(", ") || "—"}</dd>
                    <dt>Run</dt>
                    <dd class="m-mono">{card().run_id}</dd>
                  </dl>
                  <div class="m-actions">
                    <Button onClick={() => props.actions.rerun(card().run_id)} disabled={props.actions.held()}>
                      Rerun
                    </Button>
                    <Button onClick={() => props.actions.editAndRerun(card().run_id)}>Edit and rerun</Button>
                    <Button onClick={props.onOpenRun}>Open run</Button>
                  </div>
                </div>
              </Show>
            </>
          )}
        </Show>
        <Show when={card().next_stage}>
          <Button
            variant="primary"
            onClick={() => props.actions.continueRun(card().run_id)}
            disabled={props.actions.held()}
            aria-label={`Continue ${card().run_id}`}
          >
            Continue
          </Button>
        </Show>
        <Button onClick={() => props.actions.rerun(card().run_id)} disabled={props.actions.held()} aria-label={`Rerun ${card().run_id}`}>
          Rerun
        </Button>
        <Menu
          label={`More actions for ${card().run_id}`}
          icon={<EllipsisIcon />}
          items={menu()}
          onSelect={(value) => (value === "edit" ? props.actions.editAndRerun(card().run_id) : props.actions.remove(card().run_id))}
        />
      </div>
    </li>
  );
}
