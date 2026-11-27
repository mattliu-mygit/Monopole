import { createSignal, For, onSettled, Show } from "solid-js";
import { Button, EllipsisIcon, Menu, StatusDot } from "@prompting-time/ui";
import { meta, ratings } from "./format";
import { PageHeader } from "./PageHeader";
import type { Card } from "./protocol";
import { ProposalRow } from "./ProposalRow";
import type { PageStore, RunGroup } from "./store";

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

/** The status-group filters, in the order the list's chips show them. */
export const GROUPS: { group: RunGroup; label: string }[] = [
  { group: "failed", label: "Failed" },
  { group: "waiting", label: "Waiting for you" },
  { group: "review", label: "Review needed" },
  { group: "running", label: "Running" },
  { group: "done", label: "Done" },
];

/** Whether the card's title, run ID, or a judge's name contains the search text, ignoring case. */
export function matches(card: Card, query: string): boolean {
  const wanted = query.trim().toLowerCase();
  if (!wanted) return true;
  return [card.title, card.run_id, ...card.judges.map((judge) => judge.label)].some((text) => text.toLowerCase().includes(wanted));
}

/**
 * The runs, newest first, with a search over title, run ID, and judge, and a chip per status group
 * with its count. Each run is a tight card in one column: the session's first prompt, the run ID,
 * time, and turns, the judges and mean rating, and for a run that reached reflection, a Proposal
 * row. A failed run's status is a button that opens its cause in a popover; only a click opens it,
 * and a click elsewhere or Esc closes it. Each card's ⋯ menu opens, continues, reruns, edits and
 * reruns, or deletes it.
 */
export function RunList(props: { store: PageStore }) {
  const store = props.store;
  const [openRun, setOpenRun] = createSignal<string | null>(null);
  const [actionError, setActionError] = createSignal<string | null>(null);
  const actions = runActions(store, setActionError);
  const runs = () => store.state.runs?.runs ?? [];
  const searched = () => runs().filter((card) => matches(card, store.state.runQuery));
  const shown = () => searched().filter((card) => store.state.runGroup === null || card.group === store.state.runGroup);
  const count = (group: RunGroup) => searched().filter((card) => card.group === group).length;
  const setGroup = (group: RunGroup | null) => store.filterRuns(store.state.runQuery, group);

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
    <section class="m-tab">
      <PageHeader
        title="Runs"
        menu={[{ value: "refresh", label: "Read again" }]}
        onMenu={() => store.refresh()}
      >
        <Button variant="primary" onClick={() => store.open("/new")}>
          New run
        </Button>
        <Button onClick={() => void store.checkProviders()} disabled={store.state.checking || store.held()}>
          {store.state.checking ? "Checking…" : "Check providers"}
        </Button>
      </PageHeader>
      <div class="m-column">
        <div class="m-toolbar">
          <input
            class="m-search"
            type="search"
            aria-label="Search runs"
            placeholder="Search titles, run IDs, judges…"
            value={store.state.runQuery}
            onInput={(event) => store.filterRuns(event.currentTarget.value, store.state.runGroup)}
          />
          <div class="m-chips" role="group" aria-label="Filter by status">
            <button type="button" class="m-chip" aria-pressed={store.state.runGroup === null ? "true" : "false"} onClick={() => setGroup(null)}>
              All · {searched().length}
            </button>
            <For each={GROUPS}>
              {(item) => (
                <button
                  type="button"
                  class="m-chip"
                  aria-pressed={store.state.runGroup === item.group ? "true" : "false"}
                  onClick={() => setGroup(store.state.runGroup === item.group ? null : item.group)}
                >
                  {item.label} · {count(item.group)}
                </button>
              )}
            </For>
          </div>
        </div>
        <Show when={store.state.loadError}>
          {(error) => (
            <p class="m-error" role="alert">
              {error()}
            </p>
          )}
        </Show>
        <Show when={actionError()}>
          {(error) => (
            <p class="m-error" role="alert">
              {error()}
            </p>
          )}
        </Show>
        <Show when={store.state.runs} fallback={<p class="m-hint" role="status">Reading runs…</p>}>
          {(list) => (
            <Show when={list().runs.length > 0} fallback={<p class="m-hint">No runs yet. Start one from New run.</p>}>
              <Show
                when={shown().length > 0}
                fallback={
                  <p class="m-hint">
                    No runs match.{" "}
                    <button type="button" class="m-link" onClick={() => store.filterRuns("", null)}>
                      Clear the search and filter
                    </button>
                  </p>
                }
              >
                <ul class="m-runs" aria-label="Runs">
                  <For each={shown()}>
                    {(card) => (
                      <RunCard
                        card={card}
                        open={openRun() === card.run_id}
                        onToggle={() => setOpenRun(openRun() === card.run_id ? null : card.run_id)}
                        onOpenRun={() => store.open(`/runs/${encodeURIComponent(card.run_id)}`)}
                        actions={actions}
                      />
                    )}
                  </For>
                </ul>
              </Show>
              <Show when={list().truncated}>
                <p class="m-hint">Older runs are left out.</p>
              </Show>
            </Show>
          )}
        </Show>
      </div>
    </section>
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
    { value: "open", label: "Open" },
    ...(card().next_stage ? [{ value: "continue", label: "Continue", disabled: props.actions.held() }] : []),
    { value: "rerun", label: "Rerun", disabled: props.actions.held() },
    { value: "edit", label: "Edit and rerun" },
    { value: "delete", label: "Delete", disabled: !card().deletable || props.actions.held() },
  ];
  const onMenu = (value: string) => {
    const id = card().run_id;
    if (value === "open") props.onOpenRun();
    else if (value === "continue") void props.actions.continueRun(id);
    else if (value === "rerun") void props.actions.rerun(id);
    else if (value === "edit") props.actions.editAndRerun(id);
    else void props.actions.remove(id);
  };
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
        <Menu label={`More actions for ${card().run_id}`} icon={<EllipsisIcon />} items={menu()} onSelect={onMenu} />
      </div>
    </li>
  );
}
