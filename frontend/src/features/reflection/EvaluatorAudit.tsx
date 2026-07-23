import type { EvaluatorRecord } from '../../types'

function usageLabel(usage: Record<string, number>): string {
  const entries = Object.entries(usage)
  if (entries.length === 0) return 'No usage reported'
  return entries.map(([key, value]) => `${key.replaceAll('_', ' ')} ${value}`).join(' · ')
}

export default function EvaluatorAudit({
  label,
  record,
}: {
  label: 'A' | 'B'
  record: EvaluatorRecord
}) {
  return (
    <details className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs">
      <summary className="cursor-pointer font-medium text-gray-800">{label} evaluator audit</summary>
      <dl className="mt-3 grid gap-2 sm:grid-cols-2">
        <div><dt className="text-gray-500">Evaluation ID</dt><dd className="break-all font-mono">{record.evaluation_id}</dd></div>
        <div><dt className="text-gray-500">Bundle revision</dt><dd className="break-all font-mono">{record.target_revision}</dd></div>
        <div>
          <dt className="text-gray-500">Requested</dt>
          <dd>{record.requested_model} · {record.requested_family} · {record.requested_backend}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Resolved</dt>
          <dd>{record.resolved_model} · {record.resolved_family} · {record.resolved_backend}</dd>
        </div>
        <div className="sm:col-span-2"><dt className="text-gray-500">Usage</dt><dd>{usageLabel(record.usage)}</dd></div>
        <div className="sm:col-span-2"><dt className="text-gray-500">Rationale</dt><dd>{record.rationale}</dd></div>
      </dl>
    </details>
  )
}
