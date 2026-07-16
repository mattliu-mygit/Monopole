import type {
  JudgeBackendCatalog,
  ModelCatalog,
  ModelDescriptor,
  RubricCatalog,
  RunConfig,
} from '../../types'

export type ChoiceSource = 'recommended' | 'automatic' | 'overridden'
export interface Choice<T> { value: T; source: ChoiceSource }

export interface RunConfigState {
  proposalModel: Choice<string>
  judgeBackend: Choice<string>
  judgeModels: Choice<string[]>
  proposalEvaluatorModel: Choice<string>
  rubricIds: string[]
  candidateBudget: number
  force: boolean
}

export type RunConfigAction =
  | { type: 'use-recommended' }
  | { type: 'select-writer'; modelId: string }
  | { type: 'select-backend'; backend: string }
  | { type: 'select-judge'; position: number; modelId: string }
  | { type: 'remove-last-judge' }
  | { type: 'select-evaluator'; modelId: string }
  | { type: 'set-rubrics'; rubricIds: string[] }
  | { type: 'set-budget'; value: number }
  | { type: 'set-force'; value: boolean }

function supports(model: ModelDescriptor, role: ModelDescriptor['supported_roles'][number]) {
  return model.supported_roles.includes(role)
}

function backend(models: ModelCatalog, name: string): JudgeBackendCatalog | undefined {
  return models.judge_backends[name]
}

function model(source: JudgeBackendCatalog | undefined, id: string) {
  return source?.available_models.find((candidate) => candidate.id === id)
}

function writer(models: ModelCatalog, id: string) {
  return models.proposal.available_models.find((candidate) => candidate.id === id)
}

function evaluator(models: ModelCatalog, backendName: string, writerId: string): string {
  const source = backend(models, backendName)
  const writerFamily = writer(models, writerId)?.family
  const candidates = source?.proposal_evaluator_preferences
    .map((id) => model(source, id))
    .filter((candidate): candidate is ModelDescriptor =>
      candidate !== undefined && supports(candidate, 'proposal_evaluator')) ?? []
  return (candidates.find((candidate) => candidate.family !== writerFamily) ?? candidates[0])?.id ?? ''
}

function recommendedState(models: ModelCatalog, rubricIds: readonly string[]): RunConfigState {
  const backendName = models.recommended_judge_backend
  const source = backend(models, backendName)
  const writerId = models.proposal.recommended_model ?? ''
  return {
    proposalModel: { value: writerId, source: 'recommended' },
    judgeBackend: { value: backendName, source: 'recommended' },
    judgeModels: { value: [...(source?.recommended_judges ?? []).slice(0, 3)], source: 'recommended' },
    proposalEvaluatorModel: {
      value: evaluator(models, backendName, writerId),
      source: 'automatic',
    },
    rubricIds: [...rubricIds],
    candidateBudget: 3,
    force: false,
  }
}

export function initializeRunConfigState(
  models: ModelCatalog,
  rubrics: RubricCatalog,
  saved?: RunConfig | null,
): RunConfigState {
  if (!saved) return recommendedState(models, rubrics.rubrics.map((rubric) => rubric.id))
  const source = backend(models, saved.judge_backend)
  const automaticEvaluator = evaluator(models, saved.judge_backend, saved.proposal_model)
  return {
    proposalModel: {
      value: saved.proposal_model,
      source: saved.proposal_model === models.proposal.recommended_model ? 'recommended' : 'overridden',
    },
    judgeBackend: {
      value: saved.judge_backend,
      source: saved.judge_backend === models.recommended_judge_backend ? 'recommended' : 'overridden',
    },
    judgeModels: {
      value: [...saved.judge_models],
      source: saved.judge_models.length === source?.recommended_judges.length &&
        saved.judge_models.every((id, index) => id === source.recommended_judges[index])
        ? 'recommended' : 'overridden',
    },
    proposalEvaluatorModel: {
      value: saved.proposal_evaluator_model,
      source: saved.proposal_evaluator_model === automaticEvaluator ? 'automatic' : 'overridden',
    },
    rubricIds: saved.rubrics.length ? [...saved.rubrics] : rubrics.rubrics.map((rubric) => rubric.id),
    candidateBudget: saved.candidate_budget,
    force: saved.force,
  }
}

export function transitionRunConfig(
  state: RunConfigState,
  action: RunConfigAction,
  models: ModelCatalog,
): RunConfigState {
  switch (action.type) {
    case 'use-recommended': {
      const next = recommendedState(models, state.rubricIds)
      return { ...next, candidateBudget: state.candidateBudget, force: state.force }
    }
    case 'select-writer':
      return {
        ...state,
        proposalModel: { value: action.modelId, source: 'overridden' },
        proposalEvaluatorModel: state.proposalEvaluatorModel.source === 'overridden'
          ? state.proposalEvaluatorModel
          : { value: evaluator(models, state.judgeBackend.value, action.modelId), source: 'automatic' },
      }
    case 'select-backend': {
      const source = backend(models, action.backend)
      return {
        ...state,
        judgeBackend: { value: action.backend, source: 'overridden' },
        judgeModels: { value: [...(source?.recommended_judges ?? []).slice(0, 3)], source: 'recommended' },
        proposalEvaluatorModel: {
          value: evaluator(models, action.backend, state.proposalModel.value),
          source: 'automatic',
        },
      }
    }
    case 'select-judge': {
      if (action.position < 1 || action.position > 3) return state
      const values = [...state.judgeModels.value]
      if (action.position > values.length + 1) return state
      values[action.position - 1] = action.modelId
      return { ...state, judgeModels: { value: values, source: 'overridden' } }
    }
    case 'remove-last-judge':
      return state.judgeModels.value.length <= 1 ? state : {
        ...state,
        judgeModels: { value: state.judgeModels.value.slice(0, -1), source: 'overridden' },
      }
    case 'select-evaluator':
      return { ...state, proposalEvaluatorModel: { value: action.modelId, source: 'overridden' } }
    case 'set-rubrics': return { ...state, rubricIds: [...action.rubricIds] }
    case 'set-budget': return { ...state, candidateBudget: action.value }
    case 'set-force': return { ...state, force: action.value }
  }
}

export function assessRunConfig(
  state: RunConfigState,
  models: ModelCatalog,
  rubrics: RubricCatalog,
): { errors: string[]; warnings: string[] } {
  const errors: string[] = []
  const warnings: string[] = []
  const selectedWriter = writer(models, state.proposalModel.value)
  if (!selectedWriter || !supports(selectedWriter, 'proposal_writer')) {
    errors.push('Select an available proposal writer.')
  }
  const source = backend(models, state.judgeBackend.value)
  if (!source) errors.push('Select an available judge backend.')
  if (state.judgeModels.value.length < 1 || state.judgeModels.value.length > 3) {
    errors.push('Select one through three judges.')
  }
  const judges = state.judgeModels.value.map((id) => model(source, id))
  if (judges.some((judge) => judge === undefined || !supports(judge, 'judge'))) {
    errors.push('Select only available judge models.')
  }
  if (new Set(state.judgeModels.value).size !== state.judgeModels.value.length) {
    errors.push('Judge selections must be unique.')
  }
  const selectedEvaluator = model(source, state.proposalEvaluatorModel.value)
  if (!selectedEvaluator || !supports(selectedEvaluator, 'proposal_evaluator')) {
    errors.push('Select an available proposal evaluator.')
  }
  const rubricIds = new Set(rubrics.rubrics.map((rubric) => rubric.id))
  if (!state.rubricIds.length) errors.push('Select at least one rubric.')
  else if (state.rubricIds.some((id) => !rubricIds.has(id)) || new Set(state.rubricIds).size !== state.rubricIds.length) {
    errors.push('Select only available rubrics.')
  }
  if (!Number.isInteger(state.candidateBudget) || state.candidateBudget < 1) {
    errors.push('Proposal attempt limit must be at least 1.')
  } else if (state.candidateBudget > 10) {
    errors.push('Proposal attempt limit must be between 1 and 10.')
  }
  const families = new Map<string, number>()
  for (const judge of judges) if (judge && judge.family !== 'unknown') {
    families.set(judge.family, (families.get(judge.family) ?? 0) + 1)
  }
  for (const [family, count] of families) if (count > 1) {
    warnings.push(`Multiple selected judges share the ${family} model family.`)
  }
  if (selectedWriter && selectedEvaluator && selectedWriter.family === selectedEvaluator.family) {
    warnings.push(`The proposal writer and evaluator share the ${selectedWriter.family} model family.`)
  }
  return { errors, warnings }
}

export function toRunConfig(
  state: RunConfigState,
  models: ModelCatalog,
  rubrics: RubricCatalog,
): RunConfig {
  const assessment = assessRunConfig(state, models, rubrics)
  if (assessment.errors.length) throw new Error(assessment.errors[0])
  return {
    model_catalog_version: models.catalog_version,
    rubric_catalog_version: rubrics.catalog_version,
    judge_backend: state.judgeBackend.value,
    judge_models: [...state.judgeModels.value],
    proposal_model: state.proposalModel.value,
    proposal_evaluator_model: state.proposalEvaluatorModel.value,
    rubrics: [...state.rubricIds],
    candidate_budget: state.candidateBudget,
    force: state.force,
  }
}
