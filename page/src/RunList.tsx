import { createSignal, For, onSettled, Show, untrack } from "solid-js";
import { Button, EllipsisIcon, Menu, StatusDot } from "@prompting-time/ui";
import { meta, ratings } from "./format";
import { PageHeader } from "./PageHeader";
import type { Card } from "./protocol";
import { ProposalRow } from "./ProposalRow";
import type { BatchAction, BatchOutcome, PageStore, RunGroup } from "./store";

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

// A focused checkbox, such as a card's after a click, still lets the list's keys work.
const typing = (event: KeyboardEvent) =>
  event.target instanceof HTMLElement &&
  (event.target.isContentEditable ||
    ["TEXTAREA", "SELECT"].includes(event.target.tagName) ||
    (event.target instanceof HTMLInputElement && event.target.type !== "checkbox"));

const layerOpen = () =>
  [...document.querySelectorAll('[role="dialog"], [role="alertdialog"], [role="menu"]:not([hidden])')].some((element) => element.checkVisibility?.() ?? true);

const runs = (n: number) => `${n} run${n === 1 ? "" : "s"}`;

/** The notice after a batch action: what it did, then each run the server refused and why. */
export function batchNotice(action: BatchAction, outcome: BatchOutcome): string {
  const n = outcome.done.length;
  const did = action === "delete" ? `Deleted ${runs(n)}.` : action === "rerun" ? `Started ${n} rerun${n === 1 ? "" : "s"}.` : `Continued ${runs(n)}.`;
  if (outcome.failed.length === 0) return did;
  return `${did} ${outcome.failed.length} failed: ${outcome.failed.map((failure) => `${failure.item}: ${failure.error}`).join("; ")}`;
}

/**
 * The runs, newest first, with a search over title, run ID, and judge, and a chip per status group
 * with its count. Each run is a tight card in one column: the session's first prompt, the run ID,
 * time, and turns, the judges and mean rating, and for a run that reached reflection, a Proposal
 * row. A failed run's status is a button that opens its cause in a popover; only a click opens it,
 * and a click elsewhere or Esc closes it. Each card's ⋯ menu opens, continues, reruns, edits and
 * reruns, or deletes it.
 *
 * As in Whooper's review list, j and k select the next and previous run, from the first or the last
 * and around, and move the focus to its title; Enter opens it, also once the focus left the list,
 * as after coming back from the run. A click on a card outside its controls selects it too. Not
 * while typing, with a modifier, or while a popover, menu, or dialog is open, where Esc closes it.
 *
 * Runs can be checked for one action on all of them. The selected card shows its checkbox, and
 * every card does once any is checked; Shift+click checks or clears the range from the last box
 * clicked, x checks or clears the selected run, ⌘A checks every run the search and filter show,
 * and Esc clears them. A bar then says how many are checked and offers what applies to all of
 * them: Compare for exactly two, Rerun, Continue when all wait, and Delete when all may be
 * deleted. Rerun, Continue, and Delete confirm once in Prompting Time for all of them, and a
 * notice says what happened to each.
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
  // Only shown runs count, so a run deleted meanwhile drops out.
  const checked = () => shown().filter((card) => store.state.runChecked.includes(card.run_id));
  const isChecked = (runId: string) => store.state.runChecked.includes(runId);
  const [notice, setNotice] = createSignal<string | null>(null);
  // The last run whose box was clicked or toggled with x, where a Shift+click's range starts.
  let anchor: string | null = null;
  let cardList: HTMLUListElement | undefined;

  const setChecked = (runIds: string[], on: boolean) => {
    const now = untrack(() => store.state.runChecked);
    store.checkRuns(on ? [...now, ...runIds.filter((id) => !now.includes(id))] : now.filter((id) => !runIds.includes(id)));
  };
  const toggle = (runId: string, range: boolean) => {
    const cards = untrack(shown).map((card) => card.run_id);
    const on = !untrack(() => isChecked(runId));
    const from = range && anchor !== null ? cards.indexOf(anchor) : -1;
    const to = cards.indexOf(runId);
    setChecked(from < 0 ? [runId] : cards.slice(Math.min(from, to), Math.max(from, to) + 1), on);
    anchor = runId;
  };
  const batch = async (action: BatchAction) => {
    const runIds = checked().map((card) => card.run_id);
    setActionError(null);
    setNotice(null);
    try {
      const outcome = await store.runBatch(action, runIds);
      // The runs the server refused stay checked, to try again.
      store.checkRuns(outcome.failed.map((failure) => failure.item));
      setNotice(batchNotice(action, outcome));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    }
  };
  const compare = () => {
    const [a, b] = checked();
    store.open(`/compare/${encodeURIComponent(a.run_id)}/${encodeURIComponent(b.run_id)}`);
  };

  onSettled(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (!target?.closest(".m-popover, .m-status-button")) setOpenRun(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        // Esc closes the innermost thing: an open popover or menu first, then the checks.
        if (untrack(openRun) !== null) setOpenRun(null);
        else if (!event.defaultPrevented && !typing(event) && !layerOpen() && untrack(() => store.state.runChecked.length) > 0) {
          event.preventDefault();
          store.checkRuns([]);
        }
        return;
      }
      if (event.defaultPrevented || typing(event) || event.altKey) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a" && !event.shiftKey) {
        if (layerOpen()) return;
        event.preventDefault();
        store.checkRuns(untrack(shown).map((card) => card.run_id));
        return;
      }
      if (event.metaKey || event.ctrlKey) return;
      if (event.key !== "j" && event.key !== "k" && event.key !== "Enter" && event.key !== "x") return;
      if (layerOpen()) return;
      const cards = untrack(shown);
      const now = cards.findIndex((card) => card.run_id === untrack(() => store.state.runSelected));
      if (event.key === "x") {
        if (now < 0) return;
        event.preventDefault();
        toggle(cards[now].run_id, false);
        return;
      }
      if (event.key === "Enter") {
        // A focused button, the selected run's title among them, takes Enter itself.
        if (now < 0 || (document.activeElement && document.activeElement !== document.body)) return;
        event.preventDefault();
        store.open(`/runs/${encodeURIComponent(cards[now].run_id)}`);
        return;
      }
      if (cards.length === 0) return;
      event.preventDefault();
      const by = event.key === "j" ? 1 : -1;
      const next = cards[now < 0 ? (by > 0 ? 0 : cards.length - 1) : (now + by + cards.length) % cards.length];
      store.selectRun(next.run_id);
      const title = cardList?.querySelector<HTMLButtonElement>(`[data-run="${CSS.escape(next.run_id)}"] .m-title`);
      title?.focus();
      title?.scrollIntoView?.({ block: "nearest" });
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
        menu={[
          { value: "check-all", label: "Check every run shown · ⌘A" },
          { value: "refresh", label: "Read again" },
        ]}
        onMenu={(value) => (value === "check-all" ? store.checkRuns(shown().map((card) => card.run_id)) : store.refresh())}
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
        <Show when={notice()}>
          {(text) => (
            <p class="m-notice" role="status">
              <span>{text()}</span>
              <button type="button" class="m-link" onClick={() => setNotice(null)}>
                Dismiss
              </button>
            </p>
          )}
        </Show>
        <Show when={checked().length > 0}>
          <div class="m-batch" role="toolbar" aria-label="Checked runs">
            <strong>{checked().length} selected</strong>
            <Show when={checked().length === 2}>
              <Button onClick={compare}>Compare</Button>
            </Show>
            <Button onClick={() => void batch("rerun")} disabled={store.held()}>
              Rerun
            </Button>
            <Show when={checked().every((card) => card.next_stage)}>
              <Button onClick={() => void batch("continue")} disabled={store.held()}>
                Continue
              </Button>
            </Show>
            <Show when={checked().every((card) => card.deletable)}>
              <Button onClick={() => void batch("delete")} disabled={store.held()}>
                Delete
              </Button>
            </Show>
            <button type="button" class="m-link" onClick={() => store.checkRuns([])}>
              Clear · Esc
            </button>
          </div>
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
                <ul class="m-runs" aria-label="Runs" ref={(element: HTMLUListElement) => { cardList = element; }}>
                  <For each={shown()}>
                    {(card) => (
                      <RunCard
                        card={card}
                        open={openRun() === card.run_id}
                        selected={store.state.runSelected === card.run_id}
                        checkable={store.state.runSelected === card.run_id || store.state.runChecked.length > 0}
                        checked={isChecked(card.run_id)}
                        onCheck={(range) => toggle(card.run_id, range)}
                        onSelect={() => store.selectRun(card.run_id)}
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
  selected: boolean;
  /** Whether the card shows its checkbox, and whether it is checked. */
  checkable: boolean;
  checked: boolean;
  /** Checks or clears the run; `range` for Shift+click. */
  onCheck: (range: boolean) => void;
  onSelect: () => void;
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
    <li
      class="m-card"
      data-group={card().group ?? "draft"}
      data-run={card().run_id}
      data-selected={props.selected ? "" : undefined}
      data-checked={props.checked ? "" : undefined}
      onClick={(event) => {
        if (!(event.target as Element).closest("button, input, a, [role='menu'], .m-popover")) props.onSelect();
      }}
    >
      <Show when={props.checkable}>
        <input
          type="checkbox"
          class="m-check"
          aria-label={`Check ${card().run_id}`}
          checked={props.checked}
          onClick={(event) => {
            // The list's state decides; the box only draws it.
            event.preventDefault();
            props.onCheck(event.shiftKey);
          }}
        />
      </Show>
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
