import { createSignal, For, onSettled, Show } from "solid-js";
import { Button, Checkbox, Select } from "@prompting-time/ui";
import { capacitySessions, fitWords, runHints } from "./hints";
import type { ModelDescriptor, SessionEstimates, SessionSummary } from "./protocol";
import type { NewRunData, PageStore } from "./store";

const MAX_JUDGES = 3;
const NONE = "";

/**
 * New run: sessions, up to three judges, the proposal writer and evaluator, and Continue
 * automatically. Edit and rerun opens it at /new?from=<run id>, filled in with that run's
 * settings; a model that left the catalog is dropped and named. As the web app does, each model
 * option says whether it fits the selected sessions, and hints name a chosen model that doesn't
 * and models that share a family. Start run asks Prompting Time, which confirms first, and opens
 * the new run. The server checks the rest of the settings and its refusal shows here.
 */
export function NewRun(props: { store: PageStore }) {
  const store = props.store;
  return (
    <Show
      when={store.state.newRun}
      fallback={
        <Show when={store.state.loadError} fallback={<p class="m-hint" role="status">Reading sessions and models…</p>}>
          {(error) => (
            <p class="m-error" role="alert">
              {error()}
            </p>
          )}
        </Show>
      }
    >
      {(data) => <NewRunForm store={store} data={data()} />}
    </Show>
  );
}

function sessionWords(session: SessionSummary): string {
  return session.input_preview?.trim() || `Session ${session.conversation_id.slice(0, 12)}`;
}

function NewRunForm(props: { store: PageStore; data: NewRunData }) {
  const { catalog, sessions, from } = props.data;
  const models = catalog.models;
  const known = new Set(models.available_models.map((model) => model.id));
  type Role = ModelDescriptor["supported_roles"][number];
  const withRole = (role: Role) => models.available_models.filter((model) => model.supported_roles.includes(role));
  const options = (list: ModelDescriptor[], role: Role, blank?: string) => [
    ...(blank ? [{ value: NONE, label: blank }] : []),
    ...list.map((model) => {
      const fit = fitWords(catalog, model, role, sized());
      return { value: model.id, label: fit ? `${model.label} · ${fit}` : model.label };
    }),
  ];
  const saved = from?.config ?? null;
  const pick = (wanted: string | null | undefined, list: ModelDescriptor[]) =>
    wanted && known.has(wanted) ? wanted : (list[0]?.id ?? NONE);

  const startJudges = (saved?.judge_models ?? models.recommended_judges).filter((id) => known.has(id)).slice(0, MAX_JUDGES);
  const missing = saved
    ? [...saved.judge_models, saved.proposal_model, saved.proposal_evaluator_model].filter((id) => !known.has(id))
    : [];
  const [chosen, setChosen] = createSignal<string[]>([...(from?.selection?.session_ids ?? [])]);
  const [judges, setJudges] = createSignal<string[]>(Array.from({ length: MAX_JUDGES }, (_, index) => startJudges[index] ?? NONE));
  const [writer, setWriter] = createSignal(pick(saved?.proposal_model ?? models.recommended_proposal_model, withRole("proposal_writer")));
  const preferred = saved?.proposal_evaluator_model ?? models.proposal_evaluator_preferences.find((id) => known.has(id));
  const [evaluator, setEvaluator] = createSignal(pick(preferred, withRole("proposal_evaluator")));
  const [autoRun, setAutoRun] = createSignal(from?.auto_run ?? false);
  const [starting, setStarting] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  // The chosen sessions' judging token estimates; each change of sessions asks again, and only the
  // newest answer counts.
  const [estimates, setEstimates] = createSignal<SessionEstimates | null>(null);
  const [estimateError, setEstimateError] = createSignal<string | null>(null);
  let asked = 0;
  function readEstimates(ids: string[]) {
    const ask = ++asked;
    setEstimateError(null);
    if (ids.length === 0) {
      setEstimates(null);
      return;
    }
    props.store.sessionEstimates(ids).then(
      (found) => ask === asked && setEstimates(found),
      (failure) => ask === asked && setEstimateError(failure instanceof Error ? failure.message : String(failure)),
    );
  }
  const sized = () => capacitySessions(estimates(), chosen());
  const challengeJudges = (judgeModels: string[]) => {
    const challenge = (saved?.challenge_judge_models ?? models.recommended_challenge_judges).filter((id) => known.has(id));
    return challenge.length ? challenge : judgeModels.slice(0, 1);
  };
  const chosenJudges = () => [...new Set(judges().filter((id) => id !== NONE))];
  const hints = () =>
    runHints(
      catalog,
      {
        judges: chosenJudges(),
        challengeJudges: challengeJudges(chosenJudges()),
        writer: writer(),
        evaluator: evaluator(),
        rubrics: saved?.rubrics ?? [],
        candidateBudget: saved?.candidate_budget ?? 3,
        force: saved?.force ?? false,
      },
      sized(),
    );
  // Solid allows no writes while the component renders, so the first read starts once it settles.
  onSettled(() => {
    readEstimates(chosen());
  });

  // A session the old run used that the newest 50 leave out still shows, so it can stay chosen.
  const listed = new Set(sessions.sessions.map((session) => session.conversation_id));
  const rows: SessionSummary[] = [
    ...(from?.selection?.session_ids ?? [])
      .filter((id) => !listed.has(id))
      .map((id) => ({ conversation_id: id, turn_count: 0, last_activity: null, input_preview: null })),
    ...sessions.sessions,
  ];
  const toggle = (id: string) => {
    const ids = chosen().includes(id) ? chosen().filter((other) => other !== id) : [...chosen(), id];
    setChosen(ids);
    readEstimates(ids);
  };
  const setJudge = (index: number, id: string) => setJudges((ids) => ids.map((other, at) => (at === index ? id : other)));

  async function start() {
    const judgeModels = chosenJudges();
    setStarting(true);
    setError(null);
    try {
      await props.store.startRun({
        selection: {
          session_ids: chosen(),
          since: from?.selection?.since ?? null,
          until: from?.selection?.until ?? null,
          timezone: from?.selection?.timezone ?? null,
        },
        config: {
          model_catalog_version: models.catalog_version,
          rubric_catalog_version: catalog.rubrics.catalog_version,
          judge_models: judgeModels,
          challenge_judge_models: challengeJudges(judgeModels),
          proposal_model: writer(),
          proposal_evaluator_model: evaluator(),
          rubrics: saved?.rubrics ?? [],
          candidate_budget: saved?.candidate_budget ?? 3,
          force: saved?.force ?? false,
        },
        auto_run: autoRun(),
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setStarting(false);
    }
  }

  return (
    <section class="m-new-run">
      <header class="m-run-header">
        <div class="m-run-title">
          <h2>New run</h2>
          <Show when={from}>{(run) => <div class="m-meta">from {run().run_id}'s settings</div>}</Show>
        </div>
      </header>
      <Show when={missing.length > 0}>
        <p class="m-warning" role="status">
          No longer in the catalog: {missing.join(", ")}. Choose another.
        </p>
      </Show>
      <div class="m-panel">
        <h3>Sessions</h3>
        <Show when={rows.length > 0} fallback={<p class="m-hint">No sessions found in CES.</p>}>
          <div class="m-choices" role="group" aria-label="Sessions">
            <For each={rows}>
              {(session) => (
                <label>
                  <input
                    type="checkbox"
                    checked={chosen().includes(session.conversation_id)}
                    onChange={() => toggle(session.conversation_id)}
                  />
                  <span title={session.conversation_id}>{sessionWords(session)}</span>
                  <span class="m-quiet">{session.turn_count ? `${session.turn_count} turns` : ""}</span>
                </label>
              )}
            </For>
          </div>
        </Show>
        <p class="m-quiet">
          {chosen().length} selected{sessions.truncated ? " · the newest 50 sessions" : ""}
        </p>
      </div>
      <div class="m-panel">
        <h3>Models</h3>
        <div class="m-grid">
          <For each={judges()}>
            {(id, index) => (
              <Select
                label={`Judge ${index() + 1}`}
                options={options(withRole("judge"), "judge", index() === 0 ? undefined : "None")}
                value={id}
                onChange={(event) => setJudge(index(), event.currentTarget.value)}
              />
            )}
          </For>
          <Select
            label="Proposal writer"
            options={options(withRole("proposal_writer"), "proposal_writer")}
            value={writer()}
            onChange={(event) => setWriter(event.currentTarget.value)}
          />
          <Select
            label="Evaluator"
            options={options(withRole("proposal_evaluator"), "proposal_evaluator")}
            value={evaluator()}
            onChange={(event) => setEvaluator(event.currentTarget.value)}
          />
        </div>
        <Show when={hints().length > 0}>
          <ul class="m-hints" aria-label="Hints about these models">
            <For each={hints()}>{(hint) => <li class="m-warning">{hint}</li>}</For>
          </ul>
        </Show>
        <Show when={estimateError()}>{(text) => <p class="m-quiet">Couldn't read the sessions' sizes, so fits aren't shown: {text()}</p>}</Show>
      </div>
      <Show when={error()}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
      <div class="m-actions">
        <Checkbox
          label="Continue automatically (judging calls the judge models)"
          checked={autoRun()}
          onChange={(event) => setAutoRun(event.currentTarget.checked)}
        />
        <Button variant="primary" onClick={start} disabled={starting() || props.store.held() || chosen().length === 0 || judges()[0] === NONE}>
          {starting() ? "Starting…" : "Start run"}
        </Button>
      </div>
    </section>
  );
}
