// @vitest-environment jsdom

import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import UnifiedDiff from '../../src/features/reflection/UnifiedDiff'
import type { ReflectionTargetSnapshot } from '../../src/types'

afterEach(cleanup)

function target(content: string): ReflectionTargetSnapshot {
  return {
    kind: 'file',
    locator: 'CLAUDE.md',
    display_name: 'CLAUDE.md',
    path: 'CLAUDE.md',
    exists: true,
    content,
    revision: `revision:${content}`,
  }
}

describe('UnifiedDiff', () => {
  it('renders additions and removals as distinct numbered rows', () => {
    render(
      <UnifiedDiff
        locator="CLAUDE.md"
        before={target('shared\nold instruction\ntail')}
        after={target('shared\nnew instruction\nextra instruction\ntail')}
        beforeLabel="Past (B)"
        afterLabel="Proposed (C)"
      />,
    )

    const table = screen.getByRole('table', { name: 'Line changes for CLAUDE.md' })
    const removed = screen.getByText('old instruction').closest('tr')
    const added = screen.getByText('new instruction').closest('tr')
    const extra = screen.getByText('extra instruction').closest('tr')

    expect(removed?.getAttribute('data-change')).toBe('deletion')
    expect(within(removed!).getByText('2')).not.toBeNull()
    expect(within(removed!).getByLabelText('Removed line')).not.toBeNull()

    expect(added?.getAttribute('data-change')).toBe('addition')
    expect(within(added!).getByText('2')).not.toBeNull()
    expect(within(added!).getByLabelText('Added line')).not.toBeNull()

    expect(extra?.getAttribute('data-change')).toBe('addition')
    expect(within(extra!).getByText('3')).not.toBeNull()
    expect(table.textContent).not.toContain('--- CLAUDE.md')
  })

  it('labels both revisions and keeps long lines in a scrollable non-wrapping surface', () => {
    render(
      <UnifiedDiff
        locator="CLAUDE.md"
        before={target('before')}
        after={target('after')}
        beforeLabel="Past (B)"
        afterLabel="Proposed (C)"
      />,
    )

    expect(screen.getByText('Past (B)')).not.toBeNull()
    expect(screen.getByText('Proposed (C)')).not.toBeNull()
    expect(screen.getByTestId('diff-scroll').className).toContain('overflow-auto')
    expect(screen.getByText('after').className).toContain('whitespace-pre')
  })
})
