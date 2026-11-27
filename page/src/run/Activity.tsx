import { createSignal, For, onSettled, Show } from "solid-js";
import type { JSX } from "@solidjs/web";
import { clock, duration } from "./words";

/** One event in a stage's activity, as judging and reflection progress both carry them. */
export interface ActivityEvent {
  id: number;
  at: string;
  phase: string;
  message: string;
}

/**
 * A stage's activity, after the web UI's JudgingActivity and ReflectionActivity: what the stage is
 * doing now and for how long, then its events, newest last, each with its facts. It shows the
 * newest five until "Show all"; phases in `debug` (each transport attempt, say) show only after
 * "Show debug events". A failed event's words are red.
 */
export function Activity<E extends ActivityEvent>(props: {
  label: string;
  status: string;
  startedAt: string | null;
  active: boolean;
  events: readonly E[];
  facts: (event: E) => (string | null | undefined)[];
  detail?: (event: E) => JSX.Element;
  debug?: ReadonlySet<string>;
}) {
  const [all, setAll] = createSignal(false);
  const [debug, setDebug] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());
  onSettled(() => {
    const timer = setInterval(() => props.active && setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  });
  const shown = () => (debug() || !props.debug ? props.events : props.events.filter((event) => !props.debug!.has(event.phase)));
  const visible = () => (all() ? shown() : shown().slice(-5));
  const hasDebug = () => !!props.debug && props.events.some((event) => props.debug!.has(event.phase));
  const elapsed = () => {
    if (!props.startedAt) return null;
    const end = props.active ? now() : Date.parse(props.events.at(-1)?.at ?? "");
    return duration(props.startedAt, end);
  };
  const failed = (event: E) => event.phase.endsWith("_failed") || event.phase === "candidate_rejected";
  return (
    <div class="m-activity">
      <p class="m-activity-now" role="status" aria-live="polite">
        <span>{props.status}</span>
        <Show when={elapsed()}>{(text) => <span class="m-quiet">{props.active ? "running" : "took"} {text()}</span>}</Show>
      </p>
      <Show when={shown().length > 0}>
        <ol class="m-events" aria-label={props.label}>
          <For each={visible()}>
            {(event) => (
              <li aria-current={props.active && event.id === props.events.at(-1)?.id ? "step" : undefined}>
                <time class="m-mono m-quiet" datetime={event.at}>
                  {clock(event.at)}
                </time>
                <div>
                  <div class={{ "m-failed": failed(event) }}>{event.message}</div>
                  <Show when={props.facts(event).filter(Boolean).join(" · ")}>{(text) => <div class="m-quiet">{text()}</div>}</Show>
                  {props.detail?.(event)}
                </div>
              </li>
            )}
          </For>
        </ol>
        <div class="m-actions">
          <Show when={shown().length > 5}>
            <button type="button" class="m-link" aria-expanded={all() ? "true" : "false"} onClick={() => setAll(!all())}>
              {all() ? "Show the newest 5" : `Show all ${shown().length} events`}
            </button>
          </Show>
          <Show when={hasDebug()}>
            <button type="button" class="m-link" aria-pressed={debug() ? "true" : "false"} onClick={() => setDebug(!debug())}>
              {debug() ? "Hide debug events" : "Show debug events"}
            </button>
          </Show>
        </div>
      </Show>
    </div>
  );
}
