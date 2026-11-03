import type { SignalEvidence } from '../types'

interface SignalCalloutProps {
  evidence: SignalEvidence[]
}

export default function SignalCallout({ evidence }: SignalCalloutProps) {
  if (evidence.length === 0) return null

  const lowest = evidence.reduce((low, item) => (item.rating < low.rating ? item : low))
  const scorers = [...new Set(evidence.map((item) => item.signal))]

  return (
    <span
      aria-label={`Signal review recommendation: ${lowest.rating.toFixed(2)}; ${scorers.join(', ')}`}
      title={lowest.reason ?? undefined}
      className="mt-1.5 inline-flex rounded bg-amber-100 px-2 py-1 text-xs font-medium text-amber-900"
    >
      Needs review · {lowest.rating.toFixed(2)} · {scorers.join(', ')}
    </span>
  )
}
