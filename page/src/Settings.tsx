import { createMemo, createSignal, For, Show } from "solid-js";
import { Button, NumberField, TextField } from "@prompting-time/ui";
import schema from "../../schemas/plugin-settings.json";
import { PageHeader } from "./PageHeader";
import type { Settings as SettingValues } from "./protocol";
import type { PageStore } from "./store";

/** One field of schemas/plugin-settings.json, the schema Prompting Time draws its own Settings
 * form from: text, a whole number, or a group of text fields. */
interface SchemaField {
  title: string;
  description?: string;
  type: "string" | "integer" | "object";
  minimum?: number;
  properties?: Record<string, SchemaField>;
}

/** A field the user edits, by its path in the settings: `["port"]` or `["ces", "url"]`. */
interface Leaf {
  path: string[];
  key: string;
  title: string;
  description?: string;
  integer: boolean;
}

/** The fields Monopole reads each time an action runs. Monopole's server reads every other field
 * when it starts, so changing one needs a restart. */
const APPLY_NOW = new Set(["alert_webhook"]);

const groups = Object.entries(schema.properties as Record<string, SchemaField>).map(([name, field]) => {
  const leaf = (path: string[], field: SchemaField): Leaf => ({
    path,
    key: path.join("."),
    title: field.title,
    description: field.description,
    integer: field.type === "integer",
  });
  return field.type === "object"
    ? { title: field.title, description: field.description, leaves: Object.entries(field.properties ?? {}).map(([child, sub]) => leaf([name, child], sub)) }
    : { title: null, description: undefined, leaves: [leaf([name], field)] };
});
const leaves = groups.flatMap((group) => group.leaves);

const shown = (settings: SettingValues | null, leaf: Leaf): string => {
  const value = leaf.path.reduce<unknown>((at, key) => (at as Record<string, unknown> | undefined)?.[key], settings ?? undefined);
  return value === undefined || value === null ? "" : String(value);
};

/** A whole number as a number; anything else as typed, so Monopole's refusal names it. */
const parsed = (leaf: Leaf, text: string): unknown => (leaf.integer && /^\s*\d+\s*$/.test(text) ? Number(text) : text);

/** The field a refusal names: Monopole's settings errors start with the setting's key, such as
 * "port must be a whole number from 1 to 65535" or "ces.url is not a text setting". */
function fieldOf(reason: string): { key: string; text: string } | null {
  const text = reason.replace(/^Configuration error:\s*/, "");
  const key = /^([a-z_]+(?:\.[a-z_]+)?)\b/.exec(text)?.[1];
  return key && leaves.some((leaf) => leaf.key === key) ? { key, text } : null;
}

/**
 * Monopole's settings, the same ones Prompting Time's Settings tab edits: read with get-settings,
 * drawn from schemas/plugin-settings.json, and saved with save-settings, which checks them. Only
 * the changed fields are sent. A refusal shows under the field it names, and after a save that
 * changed what the server reads when it starts, the page says to restart Monopole.
 */
export function Settings(props: { store: PageStore }) {
  const store = props.store;
  const [draft, setDraft] = createSignal<Record<string, string>>({});
  const [saving, setSaving] = createSignal(false);
  const [refusal, setRefusal] = createSignal<{ key: string | null; text: string } | null>(null);
  const [saved, setSaved] = createSignal<{ restart: boolean } | null>(null);
  const value = (leaf: Leaf) => draft()[leaf.key] ?? shown(store.state.settings, leaf);
  const changed = createMemo(() => leaves.filter((leaf) => leaf.key in draft() && draft()[leaf.key] !== shown(store.state.settings, leaf)));

  function edit(leaf: Leaf, text: string) {
    setDraft((all) => ({ ...all, [leaf.key]: text }));
    setSaved(null);
    if (refusal()?.key === leaf.key || refusal()?.key === null) setRefusal(null);
  }

  async function save(event: SubmitEvent) {
    event.preventDefault();
    const fields = changed();
    if (fields.length === 0 || saving()) return;
    const changes: Record<string, unknown> = {};
    for (const leaf of fields) {
      const [first, second] = leaf.path;
      if (second === undefined) changes[first] = parsed(leaf, draft()[leaf.key]);
      else changes[first] = { ...(changes[first] as Record<string, unknown> | undefined), [second]: parsed(leaf, draft()[leaf.key]) };
    }
    setSaving(true);
    setRefusal(null);
    setSaved(null);
    try {
      await store.saveSettings(changes);
      setDraft({});
      setSaved({ restart: fields.some((leaf) => !APPLY_NOW.has(leaf.key)) });
    } catch (failure) {
      const reason = failure instanceof Error ? failure.message : String(failure);
      setRefusal(fieldOf(reason) ?? { key: null, text: reason });
    } finally {
      setSaving(false);
    }
  }

  // Each prop is read in JSX, so the field follows the draft, the refusal, and saving.
  const error = (leaf: Leaf) => (refusal()?.key === leaf.key ? refusal()!.text : undefined);
  const field = (leaf: Leaf) =>
    leaf.integer ? (
      <NumberField label={leaf.title} hint={leaf.description} error={error(leaf)} value={value(leaf)} disabled={saving()} step={1} onInput={(event) => edit(leaf, event.currentTarget.value)} />
    ) : (
      <TextField label={leaf.title} hint={leaf.description} error={error(leaf)} value={value(leaf)} disabled={saving()} onInput={(event) => edit(leaf, event.currentTarget.value)} />
    );

  return (
    <section class="m-tab m-settings">
      <PageHeader title="Settings" meta="The settings Prompting Time's Settings tab shows for Monopole" menu={[{ value: "refresh", label: "Read again" }]} onMenu={() => store.refresh()} />
      <div class="m-column">
        <Show when={store.state.loadError}>
          {(text) => (
            <p class="m-error" role="alert">
              {text()}
            </p>
          )}
        </Show>
        <Show when={store.state.settings} fallback={<Show when={!store.state.loadError}><p class="m-hint" role="status">Reading the settings…</p></Show>}>
          {/* novalidate: Monopole checks the values, so its refusal shows under the field. */}
          <form class="m-settings-form" aria-label="Monopole settings" novalidate onSubmit={(event) => void save(event)}>
            <p class="m-quiet">{schema.description}</p>
            <For each={groups}>
              {(group) => (
                <Show when={group.title} fallback={<For each={group.leaves}>{field}</For>}>
                  <fieldset class="m-panel">
                    <legend>{group.title}</legend>
                    <Show when={group.description}>{(text) => <p class="m-quiet">{text()}</p>}</Show>
                    <For each={group.leaves}>{field}</For>
                  </fieldset>
                </Show>
              )}
            </For>
            <Show when={refusal()?.key === null}>
              <p class="m-error" role="alert">
                {refusal()!.text}
              </p>
            </Show>
            <Show when={saved()}>
              {(done) => (
                <p class="m-hint" role="status">
                  {done().restart ? "Saved. Monopole's server reads these settings when it starts, so restart Monopole in Prompting Time to use them." : "Saved."}
                </p>
              )}
            </Show>
            <div class="m-actions">
              <Button disabled={changed().length === 0 || saving()} onClick={() => (setDraft({}), setRefusal(null))}>
                Revert
              </Button>
              <Button type="submit" variant="primary" disabled={changed().length === 0 || saving()}>
                {saving() ? "Saving…" : "Save"}
              </Button>
            </div>
          </form>
        </Show>
      </div>
    </section>
  );
}
