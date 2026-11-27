import { Show } from "solid-js";
import type { JSX } from "@solidjs/web";
import { EllipsisIcon, Menu, type MenuItem } from "@prompting-time/ui";

/**
 * The header every tab draws: its title and one line under it, its actions as buttons, and one ⋯
 * menu for the rest.
 */
export function PageHeader(props: {
  title: string;
  meta?: JSX.Element;
  children?: JSX.Element;
  menu: readonly MenuItem[];
  /** The ⋯ menu's name; "More <title> actions" when left out. */
  menuLabel?: string;
  onMenu: (value: string) => void;
}) {
  return (
    <header class="m-header">
      <div class="m-header-title">
        <h1>{props.title}</h1>
        <Show when={props.meta}>
          <div class="m-meta">{props.meta}</div>
        </Show>
      </div>
      <div class="m-actions">
        {props.children}
        <Menu label={props.menuLabel ?? `More ${props.title} actions`} icon={<EllipsisIcon />} items={props.menu} onSelect={props.onMenu} />
      </div>
    </header>
  );
}
