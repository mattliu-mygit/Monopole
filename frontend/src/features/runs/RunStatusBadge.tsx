import type { Run } from '../../types'

const STATUS_COLORS: Record<Run['status'], string> = {
  created: 'bg-gray-100 text-gray-700',
  scoring: 'bg-blue-100 text-blue-800',
  judging: 'bg-purple-100 text-purple-800',
  reflecting: 'bg-indigo-100 text-indigo-800',
  complete: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-yellow-100 text-yellow-800',
}

export default function RunStatusBadge({ status }: { status: Run['status'] }) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_COLORS[status]}`}
    >
      {status}
    </span>
  )
}
