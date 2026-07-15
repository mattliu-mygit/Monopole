import { describe, expect, it } from 'vitest'
import type { ReflectionTargetSnapshot } from '../../src/types'
import { lineDiff } from '../../src/features/reflection/lineDiff'

function target(content: string, exists = true): ReflectionTargetSnapshot {
  return {
    kind: 'file',
    locator: 'CLAUDE.md',
    display_name: 'CLAUDE.md',
    path: 'CLAUDE.md',
    exists,
    content: exists ? content : null,
    revision: `revision:${exists}:${content}`,
  }
}

describe('lineDiff', () => {
  it('preserves separated common lines while marking replacements', () => {
    const result = lineDiff(
      'CLAUDE.md',
      target('head\nold one\nshared\nold two\ntail'),
      target('head\nnew one\nshared\nnew two\ntail'),
    )

    expect(result).toContain('- old one')
    expect(result).toContain('+ new one')
    expect(result).toContain('  shared')
    expect(result).toContain('- old two')
    expect(result).toContain('+ new two')
  })

  it('renders a bounded truthful replacement for very large rewrites', () => {
    const before = Array.from({ length: 1_500 }, (_, index) => `old ${index}`).join('\n')
    const after = Array.from({ length: 1_500 }, (_, index) => `new ${index}`).join('\n')
    const result = lineDiff('CLAUDE.md', target(before), target(after))

    expect(result).toContain('- old 0')
    expect(result).toContain('- old 1499')
    expect(result).toContain('+ new 0')
    expect(result).toContain('+ new 1499')
    expect(result).not.toContain('  old 0')
  })

  it('represents creating an empty file through headers without inventing a blank line', () => {
    const result = lineDiff('CLAUDE.md', target('', false), target(''))

    expect(result).toBe([
      '--- /dev/null — Past (B)',
      '+++ CLAUDE.md — Proposed (C)',
    ].join('\n'))
  })

  it('represents content added to and removed from an empty file', () => {
    const added = lineDiff('CLAUDE.md', target(''), target('instruction'))
    const removed = lineDiff('CLAUDE.md', target('instruction'), target(''))

    expect(added).toContain('+ instruction')
    expect(added.split('\n')).not.toContain('- ')
    expect(removed).toContain('- instruction')
    expect(removed.split('\n')).not.toContain('+ ')
  })

  it('distinguishes a trailing newline from an unterminated final line', () => {
    const added = lineDiff('CLAUDE.md', target('instruction'), target('instruction\n'))
    const removed = lineDiff('CLAUDE.md', target('instruction\n'), target('instruction'))

    expect(added).toContain('- instruction\n\\ No newline at end of file\n+ instruction')
    expect(removed).toContain('- instruction\n+ instruction\n\\ No newline at end of file')
  })
})
