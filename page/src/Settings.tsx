import { createMemo, createSignal, For, Show } from "solid-js";
import { Button, changedSettings, fieldKey, formValue, leafFields, NumberField, settingFields, TextField, valueAt, type FormValue, type LeafField } from "@prompting-time/ui";
import schema from "../../schemas/plugin-settings.json";
import { PageHeader } from "./PageHeader";
import type { PageStore } from "./store";

// The fields of schemas/plugin-settings.json, drawn with the settings-form logic Prompting Time's
// own Settings form uses, so both read and send the same values: text, a whole number, or a group.
const fields = settingFields(JSON.stringify(schema));
const leaves = leafFields(fields);

/** The settings Monopole reads each time an action runs. Monopole's server reads every other
 * setting when it starts, so changing one needs a restart. */
const APPLY_NOW = new Set(["alert_webhook"]);

/** The field a refusal names: Monopole's settings errors start with the setting's key, such as
 * "port must be a whole number from 1 to 65535" or "ces.url is not a text setting". */
function fieldOf(reason: string): { key: string; text: string } | null {
  const text = reason.replace(/^Configuration error:\s*/, "");
  const name = /^([a-z_]+(?:\.[a-z_]+)?)\b/.exec(text)?.[1];
  const leaf = leaves.find((leaf) => leaf.path.join(".") === name);
  return leaf ? { key: fieldKey(leaf), text } : null;
}

/**
 * Monopole's settings, the same ones Prompting Time's Settings tab edits: read with get-settings,
 * drawn from schemas/plugin-settings.json, and saved with save-settings, which checks them. Only
 * the changed fields are sent. A refusal shows under the field it names, and after a save that
 * changed what the server reads when it starts, the page says to restart Monopole.
 */
export function Settings(props: { store: PageStore }) {
  const store = props.store;
  const [draft, setDraft] = createSignal<Record<string, FormValue>>({});
  const [saving, setSaving] = createSignal(false);
  const [refusal, setRefusal] = createSignal<{ key: string | null; text: string } | null>(null);
  const [saved, setSaved] = createSignal<{ restart: boolean } | null>(null);
  const value = (leaf: LeafField) => draft()[fieldKey(leaf)] ?? formValue(leaf, valueAt(store.state.settings, leaf.path));
  const changes = createMemo(() => changedSettings(fields, store.state.settings, draft()));
  const dirty = () => Object.keys(changes()).length > 0;

  function edit(leaf: LeafField, text: string) {
    setDraft((all) => ({ ...all, [fieldKey(leaf)]: text }));
    setSaved(null);
    if (refusal()?.key === fieldKey(leaf) || refusal()?.key === null) setRefusal(null);
  }

  async function save(event: SubmitEvent) {
    event.preventDefault();
    const sent = changes();
    if (!dirty() || saving()) return;
    setSaving(true);
    setRefusal(null);
    setSaved(null);
    try {
      await store.saveSettings(sent);
      setDraft({});
      setSaved({ restart: Object.keys(sent).some((key) => !APPLY_NOW.has(key)) });
    } catch (failure) {
      const reason = failure instanceof Error ? failure.message : String(failure);
      setRefusal(fieldOf(reason) ?? { key: null, text: reason });
    } finally {
      setSaving(false);
    }
  }

  // Each prop is read in JSX, so the field follows the draft, the refusal, and saving.
  const error = (leaf: LeafField) => (refusal()?.key === fieldKey(leaf) ? refusal()!.text : undefined);
  const field = (leaf: LeafField) =>
    leaf.kind === "integer" ? (
      <NumberField label={leaf.title} hint={leaf.description ?? undefined} error={error(leaf)} value={value(leaf) as string} disabled={saving()} step={1} onInput={(event) => edit(leaf, event.currentTarget.value)} />
    ) : (
      <TextField label={leaf.title} hint={leaf.description ?? undefined} error={error(leaf)} value={value(leaf) as string} disabled={saving()} onInput={(event) => edit(leaf, event.currentTarget.value)} />
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
            <For each={fields}>
              {(group) =>
                group.kind === "group" ? (
                  <fieldset class="m-panel">
                    <legend>{group.title}</legend>
                    <Show when={group.description}>{(text) => <p class="m-quiet">{text()}</p>}</Show>
                    <For each={leafFields(group.fields)}>{field}</For>
                  </fieldset>
                ) : (
                  field(group)
                )
              }
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
              <Button disabled={!dirty() || saving()} onClick={() => (setDraft({}), setRefusal(null))}>
                Revert
              </Button>
              <Button type="submit" variant="primary" disabled={!dirty() || saving()}>
                {saving() ? "Saving…" : "Save"}
              </Button>
            </div>
          </form>
        </Show>
      </div>
    </section>
  );
}
