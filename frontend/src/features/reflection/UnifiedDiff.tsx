import type { ReflectionTargetSnapshot } from '../../types'
import { lineDiff } from './lineDiff'

type ChangeKind = 'context' | 'deletion' | 'addition' | 'metadata'

interface DiffRow {
  kind: ChangeKind
  beforeLine: number | null
  afterLine: number | null
  text: string
}

function rowsFromDiff(diff: string): DiffRow[] {
  let beforeLine = 1
  let afterLine = 1

  return diff.split('\n').slice(2).map((line) => {
    if (line.startsWith('- ')) {
      const row = { kind: 'deletion' as const, beforeLine, afterLine: null, text: line.slice(2) }
      beforeLine += 1
      return row
    }
    if (line.startsWith('+ ')) {
      const row = { kind: 'addition' as const, beforeLine: null, afterLine, text: line.slice(2) }
      afterLine += 1
      return row
    }
    if (line.startsWith('  ')) {
      const row = { kind: 'context' as const, beforeLine, afterLine, text: line.slice(2) }
      beforeLine += 1
      afterLine += 1
      return row
    }
    return { kind: 'metadata' as const, beforeLine: null, afterLine: null, text: line }
  })
}

const rowClasses: Record<ChangeKind, string> = {
  context: 'bg-white text-gray-700',
  deletion: 'bg-red-50 text-red-950',
  addition: 'bg-green-50 text-green-950',
  metadata: 'bg-amber-50 text-amber-900',
}

function marker(kind: ChangeKind): { text: string; label?: string } {
  if (kind === 'deletion') return { text: '−', label: 'Removed line' }
  if (kind === 'addition') return { text: '+', label: 'Added line' }
  return { text: '' }
}

function revisionPath(locator: string, target: ReflectionTargetSnapshot | undefined): string {
  return target?.exists ? locator : '/dev/null'
}

export default function UnifiedDiff({
  locator,
  before,
  after,
  beforeLabel,
  afterLabel,
}: {
  locator: string
  before: ReflectionTargetSnapshot | undefined
  after: ReflectionTargetSnapshot | undefined
  beforeLabel: string
  afterLabel: string
}) {
  const rows = rowsFromDiff(lineDiff(locator, before, after, beforeLabel, afterLabel))

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-gray-200 bg-gray-50 px-3 py-2 text-xs">
        <div className="flex min-w-0 items-center gap-2 text-red-800">
          <span aria-hidden="true" className="font-mono font-semibold">−</span>
          <span className="font-semibold">{beforeLabel}</span>
          <span className="truncate font-mono text-gray-600">{revisionPath(locator, before)}</span>
        </div>
        <div className="flex min-w-0 items-center gap-2 text-green-800">
          <span aria-hidden="true" className="font-mono font-semibold">+</span>
          <span className="font-semibold">{afterLabel}</span>
          <span className="truncate font-mono text-gray-600">{revisionPath(locator, after)}</span>
        </div>
      </div>

      <div data-testid="diff-scroll" className="max-h-[32rem] overflow-auto">
        <table
          aria-label={`Line changes for ${locator}`}
          className="w-max min-w-full border-collapse font-mono text-xs leading-5"
        >
          <thead className="sr-only">
            <tr>
              <th>Past line</th>
              <th>Proposed line</th>
              <th>Change</th>
              <th>Content</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const changeMarker = marker(row.kind)
              return (
                <tr key={`${index}:${row.kind}:${row.text}`} data-change={row.kind} className={rowClasses[row.kind]}>
                  <td className="w-12 select-none border-r border-gray-200 px-2 text-right text-gray-400">
                    {row.beforeLine}
                  </td>
                  <td className="w-12 select-none border-r border-gray-200 px-2 text-right text-gray-400">
                    {row.afterLine}
                  </td>
                  <td
                    aria-label={changeMarker.label}
                    className="w-8 select-none px-2 text-center font-semibold"
                  >
                    {changeMarker.text}
                  </td>
                  <td className="pr-4">
                    <span className="whitespace-pre">{row.text}</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
