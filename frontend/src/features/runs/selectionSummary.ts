type SelectionRange = {
  since: string | null
  until: string | null
  timezone?: string | null
}

const dateOnlyPattern = /^\d{4}-\d{2}-\d{2}$/

function dateFormatter(timezone: string | undefined): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: timezone,
  })
}

export function formatSelectionDate(
  value: string | null,
  timezone: string | null | undefined,
): string {
  if (!value) return 'open'

  if (dateOnlyPattern.test(value)) {
    const [year, month, day] = value.split('-').map(Number)
    return dateFormatter('UTC').format(new Date(Date.UTC(year, month - 1, day, 12)))
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  try {
    return dateFormatter(timezone || undefined).format(date)
  } catch {
    return dateFormatter(undefined).format(date)
  }
}

export function formatSelectionRange(selection: SelectionRange): string {
  if (!selection.since && !selection.until) return 'All dates'
  return `${formatSelectionDate(selection.since, selection.timezone)} – ${formatSelectionDate(
    selection.until,
    selection.timezone,
  )}`
}

export function selectionTimezoneLabel(timezone: string | null | undefined): string {
  return timezone || 'Browser local time'
}
