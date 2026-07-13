import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import { getJobs } from '../api'

function statusBadgeColor(status: string): string {
  const colors: Record<string, string> = {
    complete: 'bg-green-100 text-green-800',
    running: 'bg-yellow-100 text-yellow-800',
    failed: 'bg-red-100 text-red-800',
  }
  return colors[status] ?? 'bg-gray-100 text-gray-800'
}

function typeBadgeColor(type: string): string {
  const colors: Record<string, string> = {
    score: 'bg-blue-100 text-blue-800',
    backfill: 'bg-blue-100 text-blue-800',
    judge: 'bg-purple-100 text-purple-800',
    reflect: 'bg-indigo-100 text-indigo-800',
    monitor: 'bg-teal-100 text-teal-800',
  }
  return colors[type] ?? 'bg-gray-100 text-gray-800'
}

function summarizeResult(result: any): string {
  if (!result) return '—'
  if (typeof result === 'string') return result.slice(0, 80)
  const str = JSON.stringify(result)
  return str.length > 80 ? str.slice(0, 80) + '...' : str
}

export default function Jobs() {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: 5000,
  })

  const jobs = data?.jobs ?? []

  return (
    <div>
      <PageHeader title="Jobs" />

      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {error && (
        <p className="text-red-600 text-sm">{(error as Error).message}</p>
      )}

      {jobs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b text-gray-500 text-xs uppercase tracking-wider">
                <th className="px-3 py-2 font-medium">Job ID</th>
                <th className="px-3 py-2 font-medium">Type</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Started</th>
                <th className="px-3 py-2 font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <>
                  <tr
                    key={job.job_id}
                    className="border-b hover:bg-gray-50 cursor-pointer"
                    onClick={() =>
                      setExpandedId(
                        expandedId === job.job_id ? null : job.job_id,
                      )
                    }
                  >
                    <td className="px-3 py-2 font-mono text-xs">
                      {job.job_id}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${typeBadgeColor(job.type)}`}
                      >
                        {job.type}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${statusBadgeColor(job.status)}`}
                      >
                        {job.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-500">
                      {new Date(job.started_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-gray-500 max-w-xs truncate">
                      {summarizeResult(job.result)}
                    </td>
                  </tr>
                  {expandedId === job.job_id && (
                    <tr key={`${job.job_id}-detail`} className="border-b">
                      <td colSpan={5} className="px-3 py-3">
                        {job.error ? (
                          <div className="text-red-600 text-sm mb-2">
                            Error: {job.error}
                          </div>
                        ) : null}
                        <pre className="bg-gray-50 rounded-lg border p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                          {job.result
                            ? typeof job.result === 'string'
                              ? job.result
                              : JSON.stringify(job.result, null, 2)
                            : 'No result'}
                        </pre>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!isLoading && jobs.length === 0 && (
        <p className="text-gray-500 text-sm">No jobs found.</p>
      )}
    </div>
  )
}
