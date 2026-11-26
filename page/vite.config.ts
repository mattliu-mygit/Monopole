import solid from "@solidjs/vite-plugin";
import { defineConfig } from "vitest/config";

// Prompting Time shows the page in an iframe sandboxed to scripts only, which gives it the opaque
// origin "null", so even its own module scripts load as cross-origin requests. The dev server and
// preview allow that origin; dev/index.html is that iframe with the fake host.
const sandboxed = { origin: "null" };

export default defineConfig({
  plugins: [solid()],
  // Relative asset paths, so the built page works wherever Prompting Time serves page/dist from.
  base: "./",
  server: { host: "localhost", port: 5176, strictPort: true, cors: sandboxed },
  preview: { host: "localhost", port: 5176, strictPort: true, cors: sandboxed },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    // The package's compiled components import Solid; inlining them gives tests one Solid.
    server: { deps: { inline: ["@prompting-time/ui"] } },
  },
});
