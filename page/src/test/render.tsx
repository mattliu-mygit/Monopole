import { render as renderSolid, type JSX } from "@solidjs/web";
import { flush } from "solid-js";

// As in @prompting-time/ui's tests: render into a fresh element and apply the first update;
// setup.ts unmounts it after each test.
const mounted: { dispose: () => void; container: HTMLElement }[] = [];

export function render(view: () => JSX.Element) {
  const container = document.body.appendChild(document.createElement("div"));
  mounted.push({ dispose: renderSolid(view, container), container });
  flush();
  return container;
}

export function cleanup() {
  for (const { dispose, container } of mounted.splice(0)) {
    dispose();
    container.remove();
  }
}
