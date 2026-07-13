import type {
  TurnSummary,
  TurnDetail,
  SessionSummary,
  SessionDetail,
  FeedbackItem,
  AnalysisResponse,
  Artifact,
  Rubric,
  Job,
  Run,
  DataSelection,
} from './types'

const BASE = import.meta.env.VITE_API_URL || ''

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

function qs(params: Record<string, string | number | undefined>): string {
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== '',
  )
  if (entries.length === 0) return ''
  return '?' + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString()
}

export function getTurns(
  params?: { since?: string; limit?: number },
): Promise<{ turns: TurnSummary[] }> {
  return apiFetch(`/api/turns${qs({ since: params?.since, limit: params?.limit })}`)
}

export function getTurn(traceId: string): Promise<TurnDetail> {
  return apiFetch(`/api/turns/${encodeURIComponent(traceId)}`)
}

export function getSessions(
  params?: { since?: string; limit?: number },
): Promise<{ sessions: SessionSummary[] }> {
  return apiFetch(`/api/sessions${qs({ since: params?.since, limit: params?.limit })}`)
}

export function getSession(conversationId: string): Promise<SessionDetail> {
  return apiFetch(`/api/sessions/${encodeURIComponent(conversationId)}`)
}

export function getFeedback(
  params: { trace_id?: string; ref?: string },
): Promise<{ feedback: FeedbackItem[] }> {
  return apiFetch(`/api/feedback${qs({ trace_id: params.trace_id, ref: params.ref })}`)
}

export function getAnalysis(
  params?: { limit?: number },
): Promise<AnalysisResponse> {
  return apiFetch(`/api/analyze${qs({ limit: params?.limit })}`)
}

export function getArtifacts(): Promise<{ artifacts: Artifact[] }> {
  return apiFetch('/api/artifacts')
}

export function getRubrics(): Promise<{ rubrics: Rubric[] }> {
  return apiFetch('/api/rubrics')
}

export function getJobs(): Promise<{ jobs: Job[] }> {
  return apiFetch('/api/jobs')
}

export function getJob(jobId: string): Promise<Job> {
  return apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`)
}

export function submitJob(
  type: string,
  params: Record<string, unknown>,
): Promise<{ job_id: string; status: string }> {
  return apiFetch(`/api/jobs/${type}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
}

export function applyReflection(
  jobId: string,
): Promise<{ applied: string[] }> {
  return apiFetch('/api/jobs/reflect/apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  })
}

export function getModels(): Promise<
  Record<string, { default: string[]; poll: string[]; escalation: string }>
> {
  return apiFetch('/api/models')
}

export function getMonitorState(): Promise<{ seen_keys: string[]; count: number }> {
  return apiFetch('/api/monitor/state')
}

export function getRuns(): Promise<{ runs: Run[] }> {
  return apiFetch('/api/runs')
}

export function getRun(runId: string): Promise<Run> {
  return apiFetch(`/api/runs/${encodeURIComponent(runId)}`)
}

export function createRun(): Promise<Run> {
  return apiFetch('/api/runs', { method: 'POST' })
}

export function setRunSelection(
  runId: string,
  selection: Partial<DataSelection>,
): Promise<Run> {
  return apiFetch(`/api/runs/${encodeURIComponent(runId)}/selection`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(selection),
  })
}

export function advanceRun(
  runId: string,
  params?: Record<string, unknown>,
): Promise<Run> {
  return apiFetch(`/api/runs/${encodeURIComponent(runId)}/advance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params ?? {}),
  })
}

// NOTE: there is no backend endpoint for this yet. The evaluation-runs plan
// (docs/plans/2026-07-12-evaluation-runs.md, Task 4) assumes a "run-scoped
// apply" endpoint exists to replace `/api/jobs/reflect/apply`, but no task in
// that plan actually adds one (Task 2 only shipped create/list/get/selection/
// advance). This calls the RESTful shape that fits the rest of the run API;
// it will 404 until a matching `POST /api/runs/{run_id}/apply` endpoint is
// added to api.py. Surfaced as a concern in task-3-report.md.
export function applyRunReflection(runId: string): Promise<{ applied: string[] }> {
  return apiFetch(`/api/runs/${encodeURIComponent(runId)}/apply`, {
    method: 'POST',
  })
}
