import { createFakeHost } from "../src/fakeHost";

// Shows the page as Prompting Time will: in an iframe sandboxed to scripts only, so the page talks to
// this window through postMessage alone, here with the fake host. ?route=/runs/<run id> or
// ?route=/providers opens that route once the page is ready.
const frame = document.createElement("iframe");
frame.title = "Monopole";
frame.sandbox.add("allow-scripts");
frame.src = "/index.html";
document.body.append(frame);

const route = new URLSearchParams(location.search).get("route") ?? undefined;
const host = createFakeHost((message) => frame.contentWindow?.postMessage(message, "*"), {
  confirm: (runId) => window.confirm(`Delete ${runId}?`),
  openOnReady: route,
});
window.addEventListener("message", (event) => {
  if (event.source === frame.contentWindow) host.receive(event.data);
});
