import "@prompting-time/ui/styles.css";
import "./app.css";
import { onSettled } from "solid-js";
import { render } from "@solidjs/web";
import { App } from "./App";
import { createPageStore, parentLink } from "./store";

render(() => {
  const store = createPageStore(parentLink());
  onSettled(() => {
    store.ready();
    return store.dispose;
  });
  return <App store={store} />;
}, document.getElementById("root")!);
