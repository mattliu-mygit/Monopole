// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Dialog from '../../src/components/Dialog'

afterEach(cleanup)

describe('Dialog', () => {
  it('focuses the safe action and restores focus to the opener when closed', async () => {
    function Harness() {
      const [open, setOpen] = useState(false)
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>Open dialog</button>
          {open && (
            <Dialog title="Confirm change" onClose={() => setOpen(false)}>
              <button type="button">Apply change</button>
            </Dialog>
          )}
        </>
      )
    }

    render(<Harness />)
    const opener = screen.getByRole('button', { name: 'Open dialog' })
    opener.focus()
    fireEvent.click(opener)
    const cancel = screen.getByRole('button', { name: 'Cancel' })

    await waitFor(() => expect(document.activeElement).toBe(cancel))
    fireEvent.click(cancel)
    await waitFor(() => expect(document.activeElement).toBe(opener))
  })

  it('cycles Tab and Shift+Tab through links, inputs, textareas, and buttons', async () => {
    render(
      <Dialog title="Keyboard dialog" onClose={() => undefined}>
        <a href="/details">Details</a>
        <input aria-label="Name" />
        <textarea aria-label="Notes" />
        <button type="button">Apply</button>
      </Dialog>,
    )

    const order = [
      screen.getByRole('link', { name: 'Details' }),
      screen.getByRole('textbox', { name: 'Name' }),
      screen.getByRole('textbox', { name: 'Notes' }),
      screen.getByRole('button', { name: 'Apply' }),
      screen.getByRole('button', { name: 'Cancel' }),
    ]
    await waitFor(() => expect(document.activeElement).toBe(order.at(-1)))

    order[0].focus()
    for (let index = 1; index < order.length; index += 1) {
      fireEvent.keyDown(document, { key: 'Tab' })
      expect(document.activeElement).toBe(order[index])
    }
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(document.activeElement).toBe(order[0])
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(order.at(-1))
  })

  it('skips untabbable, hidden, inert, aria-hidden, and disabled-fieldset controls', async () => {
    render(
      <Dialog title="Filtered dialog" onClose={() => undefined}>
        <a href="/details">Details</a>
        <button type="button" tabIndex={-1}>Untabbable</button>
        <button type="button" hidden>Hidden control</button>
        <div hidden><button type="button">Hidden ancestor control</button></div>
        <div style={{ display: 'none' }}><button type="button">Display none control</button></div>
        <div style={{ visibility: 'hidden' }}><button type="button">Invisible control</button></div>
        <div inert><button type="button">Inert control</button></div>
        <div aria-hidden="true"><button type="button">Aria hidden control</button></div>
        <fieldset disabled><button type="button">Fieldset-disabled control</button></fieldset>
        <button type="button">Apply</button>
      </Dialog>,
    )
    const details = screen.getByRole('link', { name: 'Details' })
    const apply = screen.getByRole('button', { name: 'Apply' })
    const cancel = screen.getByRole('button', { name: 'Cancel' })
    await waitFor(() => expect(document.activeElement).toBe(cancel))

    details.focus()
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(document.activeElement).toBe(apply)
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(details)
  })

  it('focuses the first tabbable child when initially busy', async () => {
    render(
      <Dialog title="Initially busy dialog" onClose={() => undefined} busy>
        <a href="/progress">View progress</a>
        <button type="button" disabled>Apply</button>
      </Dialog>,
    )

    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('link', { name: 'View progress' })),
    )
  })

  it('uses the dialog container when initially busy with no tabbable children', async () => {
    render(
      <Dialog title="All disabled dialog" onClose={() => undefined} busy>
        <button type="button" disabled>Apply</button>
      </Dialog>,
    )
    const dialog = screen.getByRole('dialog', { name: 'All disabled dialog' })

    await waitFor(() => expect(document.activeElement).toBe(dialog))
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(document.activeElement).toBe(dialog)
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(dialog)
  })

  it('moves focus to the container when a busy transition disables every child', async () => {
    const { rerender } = render(
      <Dialog title="Transition dialog" onClose={() => undefined}>
        <button type="button">Apply</button>
      </Dialog>,
    )
    const apply = screen.getByRole('button', { name: 'Apply' })
    apply.focus()

    rerender(
      <Dialog title="Transition dialog" onClose={() => undefined} busy>
        <button type="button" disabled>Apply</button>
      </Dialog>,
    )
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(document.activeElement).toBe(screen.getByRole('dialog', { name: 'Transition dialog' }))
  })

  it('closes with Escape only when it is not busy', () => {
    const onClose = vi.fn()
    const { rerender } = render(
      <Dialog title="Escape dialog" onClose={onClose}>
        <button type="button">Continue</button>
      </Dialog>,
    )

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)

    rerender(
      <Dialog title="Escape dialog" onClose={onClose} busy>
        <button type="button">Continue</button>
      </Dialog>,
    )
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('closes from backdrop clicks only when it is not busy', () => {
    const onClose = vi.fn()
    const { rerender } = render(
      <Dialog title="Backdrop dialog" onClose={onClose}>
        <button type="button">Continue</button>
      </Dialog>,
    )
    const backdrop = screen.getByRole('dialog').parentElement!

    fireEvent.click(screen.getByRole('dialog'))
    expect(onClose).not.toHaveBeenCalled()
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalledTimes(1)

    rerender(
      <Dialog title="Backdrop dialog" onClose={onClose} busy>
        <button type="button">Continue</button>
      </Dialog>,
    )
    fireEvent.click(screen.getByRole('dialog').parentElement!)
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
