import { onSettled } from "solid-js";
import { App } from "../App";
import { createFakeHost, type FakeHostOptions } from "../fakeHost";
import { createPageStore, type HostLink } from "../store";
import { render } from "./render";

/**
 * Renders the page wired to the fake host. Messages go both ways through structuredClone in a
 * microtask, as postMessage delivers them, so a message that would not survive postMessage fails here.
 */
export function renderWithFakeHost(options: FakeHostOptions = {}) {
  let toPage: ((data: unknown) => void) | undefined;
  const host = createFakeHost((message) => queueMicrotask(() => toPage?.(structuredClone(message))), options);
  const link: HostLink = {
    post: (message) => queueMicrotask(() => host.receive(structuredClone(message))),
    listen: (receive) => {
      toPage = receive;
      return () => (toPage = undefined);
    },
  };
  render(() => {
    const store = createPageStore(link);
    onSettled(() => {
      store.ready();
      return store.dispose;
    });
    return <App store={store} />;
  });
  return host;
}
