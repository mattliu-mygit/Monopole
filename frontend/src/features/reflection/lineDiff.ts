import type { ReflectionTargetSnapshot } from '../../types'

interface DiffLine {
  text: string
  terminated: boolean
}

function lines(target: ReflectionTargetSnapshot | undefined): DiffLine[] {
  if (!target?.exists) return []
  const content = target.content ?? ''
  if (content === '') return []

  const terminated = content.endsWith('\n')
  const values = content.split('\n')
  if (terminated) values.pop()
  return values.map((text, index) => ({
    text,
    terminated: index < values.length - 1 || terminated,
  }))
}

function sameLine(left: DiffLine, right: DiffLine): boolean {
  return left.text === right.text && left.terminated === right.terminated
}

function renderLine(prefix: ' ' | '-' | '+', line: DiffLine): string[] {
  return [
    `${prefix} ${line.text}`,
    ...(!line.terminated ? ['\\ No newline at end of file'] : []),
  ]
}

// The exact LCS table is useful for ordinary instruction files but grows with
// the product of both line counts. A truthful remove/add block is preferable to
// freezing the review page on an unexpectedly large or completely rewritten
// managed file.
const MAX_LCS_CELLS = 2_000_000

function diffLines(before: DiffLine[], after: DiffLine[]): string[] {
  let prefixLength = 0
  while (
    prefixLength < before.length &&
    prefixLength < after.length &&
    sameLine(before[prefixLength], after[prefixLength])
  ) {
    prefixLength += 1
  }

  let suffixLength = 0
  while (
    suffixLength < before.length - prefixLength &&
    suffixLength < after.length - prefixLength &&
    sameLine(
      before[before.length - 1 - suffixLength],
      after[after.length - 1 - suffixLength],
    )
  ) {
    suffixLength += 1
  }

  const prefix = before.slice(0, prefixLength).flatMap((line) => renderLine(' ', line))
  const suffix = suffixLength === 0
    ? []
    : before.slice(before.length - suffixLength).flatMap((line) => renderLine(' ', line))
  const oldMiddle = before.slice(prefixLength, before.length - suffixLength)
  const newMiddle = after.slice(prefixLength, after.length - suffixLength)
  if (oldMiddle.length * newMiddle.length > MAX_LCS_CELLS) {
    return [
      ...prefix,
      ...oldMiddle.flatMap((line) => renderLine('-', line)),
      ...newMiddle.flatMap((line) => renderLine('+', line)),
      ...suffix,
    ]
  }
  const lengths = Array.from(
    { length: oldMiddle.length + 1 },
    () => new Uint32Array(newMiddle.length + 1),
  )

  for (let oldIndex = oldMiddle.length - 1; oldIndex >= 0; oldIndex -= 1) {
    for (let newIndex = newMiddle.length - 1; newIndex >= 0; newIndex -= 1) {
      lengths[oldIndex][newIndex] = sameLine(oldMiddle[oldIndex], newMiddle[newIndex])
        ? lengths[oldIndex + 1][newIndex + 1] + 1
        : Math.max(lengths[oldIndex + 1][newIndex], lengths[oldIndex][newIndex + 1])
    }
  }

  const result: string[] = []
  let oldIndex = 0
  let newIndex = 0
  while (oldIndex < oldMiddle.length || newIndex < newMiddle.length) {
    if (
      oldIndex < oldMiddle.length &&
      newIndex < newMiddle.length &&
      sameLine(oldMiddle[oldIndex], newMiddle[newIndex])
    ) {
      result.push(...renderLine(' ', oldMiddle[oldIndex]))
      oldIndex += 1
      newIndex += 1
    } else if (
      oldIndex < oldMiddle.length &&
      (newIndex >= newMiddle.length || lengths[oldIndex + 1][newIndex] >= lengths[oldIndex][newIndex + 1])
    ) {
      result.push(...renderLine('-', oldMiddle[oldIndex]))
      oldIndex += 1
    } else {
      result.push(...renderLine('+', newMiddle[newIndex]))
      newIndex += 1
    }
  }
  return [...prefix, ...result, ...suffix]
}

export function lineDiff(
  locator: string,
  before: ReflectionTargetSnapshot | undefined,
  after: ReflectionTargetSnapshot | undefined,
  beforeLabel = 'Past (A)',
  afterLabel = 'Proposed (B)',
): string {
  return [
    `--- ${before?.exists ? locator : '/dev/null'} — ${beforeLabel}`,
    `+++ ${after?.exists ? locator : '/dev/null'} — ${afterLabel}`,
    ...diffLines(lines(before), lines(after)),
  ].join('\n')
}
