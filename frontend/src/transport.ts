/** Generated HTTP transport contracts. Do not hand-copy API payload shapes here. */
import type { components, paths } from './generated/api'

export type AnalysisTransport = components['schemas']['AnalysisResponse']
export type ModelCatalogTransport = components['schemas']['ModelCatalogResponse']
export type RubricCatalogTransport = components['schemas']['RubricCatalog']
export type RunTransport = components['schemas']['RunResponse']
export type RunListTransport = components['schemas']['RunListResponse']
export type SessionListTransport = components['schemas']['SessionListResponse']
export type SessionDetailTransport = components['schemas']['SessionDetailResponse']

export type PromoteRequestTransport =
  paths['/api/runs/{run_id}/promote']['post']['requestBody']['content']['application/json']
