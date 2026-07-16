import type {
  SessionSummary,
  SessionDetail,
  Run,
  RunSummary,
  ModelCatalog,
  RubricCatalog,
  DataSelection,
  RunConfig,
} from './types'
import type {
  AnalysisTransport,
  ModelCatalogTransport,
  PromoteRequestTransport,
  RubricCatalogTransport,
  RunListTransport,
  RunTransport,
  SessionDetailTransport,
  SessionListTransport,
} from './transport'

const BASE = import.meta.env.VITE_API_URL || ''

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, body: string, detail: unknown) {
    const message =
      detail && typeof detail === 'object' && 'message' in detail &&
      typeof detail.message === 'string'
        ? detail.message
        : body
    super(`API ${status}: ${message}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    let detail: unknown = body
    try {
      const parsed = JSON.parse(body) as { detail?: unknown }
      detail = parsed.detail ?? parsed
    } catch {
      // Keep the raw body when an upstream returns plain text.
    }
    throw new ApiError(res.status, body, detail)
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

export function getSessions(
  params?: { since?: string; until?: string; timezone?: string; limit?: number },
): Promise<{
  sessions: SessionSummary[]
  total?: number
  truncated?: boolean
  limit?: number
}> {
  return apiFetch<SessionListTransport>(
    `/api/sessions${qs({
      since: params?.since,
      until: params?.until,
      timezone: params?.timezone,
      limit: params?.limit,
    })}`,
  )
}

export function getSession(conversationId: string): Promise<SessionDetail> {
  return apiFetch<SessionDetailTransport>(
    `/api/sessions/${encodeURIComponent(conversationId)}`,
  )
}

export function getAnalysis(
  params?: { limit?: number },
): Promise<AnalysisTransport> {
  return apiFetch<AnalysisTransport>(`/api/analyze${qs({ limit: params?.limit })}`)
}

export function getRubrics(): Promise<RubricCatalog> {
  return apiFetch<RubricCatalogTransport>('/api/rubrics')
}

export function getModels(): Promise<ModelCatalog> {
  return apiFetch<ModelCatalogTransport>('/api/models')
}

export function getRuns(): Promise<{ runs: RunSummary[] }> {
  return apiFetch<RunListTransport>('/api/runs')
}

export function getRun(runId: string): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}`)
}

export function createRun(): Promise<Run> {
  return apiFetch<RunTransport>('/api/runs', {
    method: 'POST',
  })
}

export function setRunSelection(
  runId: string,
  selection: DataSelection,
): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/selection`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(selection),
  })
}

export function setRunConfig(
  runId: string,
  config: RunConfig,
): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function setAutoRun(runId: string, autoRun: boolean): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/auto_run`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auto_run: autoRun }),
  })
}

export function cancelRun(runId: string): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
  })
}

export function advanceRun(runId: string): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/advance`, {
    method: 'POST',
  })
}

export function setReflectionSelection(
  runId: string,
  candidateId: string,
  expectedRevision = 0,
  discardDraft = false,
): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/reflection_selection`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      candidate_id: candidateId,
      expected_revision: expectedRevision,
      discard_draft: discardDraft,
    }),
  })
}

export function saveReflectionDraft(
  runId: string,
  contents: Record<string, string | null>,
  expectedRevision: number,
  expectedDraftRevision: string | null,
): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/reflection_draft`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_revision: expectedRevision,
      expected_draft_revision: expectedDraftRevision,
      contents,
    }),
  })
}

export function resetReflectionDraft(
  runId: string,
  expectedRevision: number,
  expectedDraftRevision: string | null,
): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/reflection_draft`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_revision: expectedRevision,
      expected_draft_revision: expectedDraftRevision,
    }),
  })
}

export function promoteRunReflection(
  runId: string,
  options: {
    expectedRevision: number
    expectedDraftRevision?: string | null
    idempotencyKey: string
    acknowledgeUnevaluated: boolean
  },
): Promise<Run> {
  const body: PromoteRequestTransport = {
    expected_revision: options.expectedRevision,
    expected_draft_revision: options.expectedDraftRevision,
    idempotency_key: options.idempotencyKey,
    acknowledge_unevaluated: options.acknowledgeUnevaluated,
  }
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function dismissRunReflection(runId: string, expectedRevision: number): Promise<Run> {
  return apiFetch<RunTransport>(`/api/runs/${encodeURIComponent(runId)}/dismiss`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected_revision: expectedRevision }),
  })
}
