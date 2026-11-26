import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "./render";

afterEach(cleanup);

// jsdom has no ResizeObserver, which the package's Zag.js components observe sizes with.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
