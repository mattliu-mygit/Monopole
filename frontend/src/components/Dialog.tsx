import { useEffect, useId, useRef, type ReactNode } from 'react'

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'input',
  'select',
  'textarea',
  'button',
  '[tabindex]',
].join(', ')

function isTabbable(element: HTMLElement, dialog: HTMLElement): boolean {
  if (element.tabIndex < 0 || element.matches(':disabled')) return false

  let current: HTMLElement | null = element
  while (current) {
    if (
      current.hidden ||
      current.hasAttribute('inert') ||
      current.getAttribute('aria-hidden') === 'true'
    ) {
      return false
    }
    const style = window.getComputedStyle(current)
    if (style.display === 'none' || style.visibility === 'hidden') return false
    if (current === dialog) break
    current = current.parentElement
  }
  return true
}

function tabbableChildren(dialog: HTMLElement): HTMLElement[] {
  return Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter((element) =>
    isTabbable(element, dialog),
  )
}

interface DialogProps {
  title: string
  children: ReactNode
  onClose: () => void
  busy?: boolean
  closeLabel?: string
}

export default function Dialog({
  title,
  children,
  onClose,
  busy = false,
  closeLabel = 'Cancel',
}: DialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const previouslyFocused = useRef<HTMLElement | null>(null)
  const onCloseRef = useRef(onClose)
  const busyRef = useRef(busy)
  const titleId = useId()

  onCloseRef.current = onClose
  busyRef.current = busy

  useEffect(() => {
    previouslyFocused.current = document.activeElement as HTMLElement | null
    const dialog = dialogRef.current
    if (!dialog) return
    const items = tabbableChildren(dialog)
    const safeAction = closeButtonRef.current
    const initialFocus = safeAction && isTabbable(safeAction, dialog)
      ? safeAction
      : items[0] ?? dialog
    initialFocus.focus()

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (!busyRef.current) onCloseRef.current()
        return
      }
      if (event.key !== 'Tab') return

      const currentDialog = dialogRef.current
      if (!currentDialog) return
      const items = tabbableChildren(currentDialog)
      event.preventDefault()
      if (items.length === 0) {
        currentDialog.focus()
        return
      }
      const currentIndex = items.indexOf(document.activeElement as HTMLElement)
      const nextIndex = event.shiftKey
        ? currentIndex <= 0 ? items.length - 1 : currentIndex - 1
        : currentIndex < 0 || currentIndex === items.length - 1 ? 0 : currentIndex + 1
      items[nextIndex].focus()
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      previouslyFocused.current?.focus()
    }
  }, [])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(event) => {
        if (event.target === event.currentTarget && !busyRef.current) onCloseRef.current()
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="w-full max-w-lg rounded-xl bg-white p-6 shadow-xl"
      >
        <h2 id={titleId} className="text-lg font-semibold text-gray-900">{title}</h2>
        {children}
        <button
          ref={closeButtonRef}
          type="button"
          disabled={busy}
          onClick={onClose}
          className="mt-4 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {closeLabel}
        </button>
      </div>
    </div>
  )
}
