import type { SignalEvidence } from '../types'

interface SignalCalloutProps {
  evidence: SignalEvidence[]
}

function signalLabel(signal: string): string {
  return signal.replaceAll('-', ' ')
}

export default function SignalCallout({ evidence }: SignalCalloutProps) {
  if (evidence.length === 0) return null

  const lowestRating = Math.min(...evidence.map((item) => item.rating))
  const signals = [...new Set(evidence.map((item) => signalLabel(item.signal)))]

  return (
    <span
      aria-label={`Signal review recommendation: ${lowestRating.toFixed(2)}; ${signals.join(', ')}`}
      className="mt-1.5 inline-flex rounded bg-amber-100 px-2 py-1 text-xs font-medium text-amber-900"
    >
      Needs review · {lowestRating.toFixed(2)} · {signals.join(', ')}
    </span>
  )
}
