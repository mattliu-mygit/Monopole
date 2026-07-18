import type {
  ModelCatalog,
  ModelDescriptor,
  RubricCatalog,
  RunConfig,
  SessionDetail,
} from '../../types'
import { estimateModelCapacity } from './modelCapacity'

type SessionTokens = Pick<SessionDetail, 'judging_token_estimates'>

export type ChoiceSource = 'recommended' | 'automatic' | 'overridden'
export interface Choice<T> { value: T; source: ChoiceSource }

export interface RunConfigState {
  proposalModel: Choice<string>
  judgeModels: Choice<string[]>
  challengeJudgeModels: Choice<string[]>
  proposalEvaluatorModel: Choice<string>
  rubricIds: string[]
  candidateBudget: number
  force: boolean
}

export type RunConfigAction =
  | { type: 'use-recommended' }
  | { type: 'select-writer'; modelId: string }
  | { type: 'select-judge'; position: number; modelId: string }
  | { type: 'remove-last-judge' }
  | { type: 'select-challenge-judge'; position: number; modelId: string }
  | { type: 'remove-last-challenge-judge' }
  | { type: 'select-evaluator'; modelId: string }
  | { type: 'set-rubrics'; rubricIds: string[] }
  | { type: 'set-budget'; value: number }
  | { type: 'set-force'; value: boolean }

function supports(model: ModelDescriptor, role: ModelDescriptor['supported_roles'][number]) {
  return model.supported_roles.includes(role)
}

function model(models: ModelCatalog, id: string) {
  return models.available_models.find((candidate) => candidate.id === id)
}

function fits(
  candidate: ModelDescriptor,
  models: ModelCatalog,
  sessions: readonly SessionTokens[],
): boolean {
  return estimateModelCapacity(candidate, sessions, models.judging_context).fits !== false
}

function evaluator(
  models: ModelCatalog,
  writerId: string,
  sessions: readonly SessionTokens[] = [],
): string {
  const writerFamily = model(models, writerId)?.family
  const candidates = models.proposal_evaluator_preferences
    .map((id) => model(models, id))
    .filter((candidate): candidate is ModelDescriptor =>
      candidate !== undefined
      && supports(candidate, 'proposal_evaluator')
      && fits(candidate, models, sessions))
  return (candidates.find((candidate) => candidate.family !== writerFamily) ?? candidates[0])?.id ?? ''
}

function same(values: readonly string[], recommended: readonly string[]): boolean {
  return values.length === recommended.length &&
    values.every((value, index) => value === recommended[index])
}

function fittingIds(
  models: ModelCatalog,
  role: ModelDescriptor['supported_roles'][number],
  preferred: readonly string[],
  sessions: readonly SessionTokens[],
): string[] {
  return [...preferred, ...models.available_models.map((candidate) => candidate.id)]
    .filter((id, index, values) => values.indexOf(id) === index)
    .filter((id) => {
      const candidate = model(models, id)
      return candidate !== undefined
        && supports(candidate, role)
        && fits(candidate, models, sessions)
    })
}

function recommendedState(
  models: ModelCatalog,
  rubricIds: readonly string[],
  sessions: readonly SessionTokens[] = [],
): RunConfigState {
  const writerId = fittingIds(
    models,
    'proposal_writer',
    models.recommended_proposal_model ? [models.recommended_proposal_model] : [],
    sessions,
  )[0] ?? ''
  const judgeIds = fittingIds(models, 'judge', models.recommended_judges, sessions)
  const challengeJudgeIds = fittingIds(
    models,
    'judge',
    models.recommended_challenge_judges,
    sessions,
  )
  return {
    proposalModel: { value: writerId, source: 'recommended' },
    judgeModels: {
      value: judgeIds.slice(0, Math.max(1, models.recommended_judges.length)),
      source: 'recommended',
    },
    challengeJudgeModels: {
      value: challengeJudgeIds.slice(
        0,
        Math.max(1, models.recommended_challenge_judges.length),
      ),
      source: 'recommended',
    },
    proposalEvaluatorModel: {
      value: evaluator(models, writerId, sessions),
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
  const automaticEvaluator = evaluator(models, saved.proposal_model)
  return {
    proposalModel: {
      value: saved.proposal_model,
      source: saved.proposal_model === models.recommended_proposal_model ? 'recommended' : 'overridden',
    },
    judgeModels: {
      value: [...saved.judge_models],
      source: same(saved.judge_models, models.recommended_judges) ? 'recommended' : 'overridden',
    },
    challengeJudgeModels: {
      value: [...saved.challenge_judge_models],
      source: same(saved.challenge_judge_models, models.recommended_challenge_judges)
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
  sessions: readonly SessionTokens[] = [],
): RunConfigState {
  switch (action.type) {
    case 'use-recommended': {
      const next = recommendedState(models, state.rubricIds, sessions)
      return { ...next, candidateBudget: state.candidateBudget, force: state.force }
    }
    case 'select-writer':
      return {
        ...state,
        proposalModel: { value: action.modelId, source: 'overridden' },
        proposalEvaluatorModel: state.proposalEvaluatorModel.source === 'overridden'
          ? state.proposalEvaluatorModel
          : { value: evaluator(models, action.modelId, sessions), source: 'automatic' },
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
    case 'select-challenge-judge': {
      if (action.position < 1 || action.position > 3) return state
      const values = [...state.challengeJudgeModels.value]
      if (action.position > values.length + 1) return state
      values[action.position - 1] = action.modelId
      return { ...state, challengeJudgeModels: { value: values, source: 'overridden' } }
    }
    case 'remove-last-challenge-judge':
      return state.challengeJudgeModels.value.length <= 1 ? state : {
        ...state,
        challengeJudgeModels: {
          value: state.challengeJudgeModels.value.slice(0, -1),
          source: 'overridden',
        },
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
  sessions: readonly SessionTokens[] = [],
): { errors: string[]; warnings: string[] } {
  const errors: string[] = []
  const warnings: string[] = []
  const selectedWriter = model(models, state.proposalModel.value)
  if (!selectedWriter || !supports(selectedWriter, 'proposal_writer')) {
    errors.push('Select an available proposal writer.')
  }
  if (state.judgeModels.value.length < 1 || state.judgeModels.value.length > 3) {
    errors.push('Select one through three judges.')
  }
  const judges = state.judgeModels.value.map((id) => model(models, id))
  if (judges.some((judge) => judge === undefined || !supports(judge, 'judge'))) {
    errors.push('Select only available judge models.')
  }
  if (new Set(state.judgeModels.value).size !== state.judgeModels.value.length) {
    errors.push('Judge selections must be unique.')
  }
  if (state.challengeJudgeModels.value.length < 1 || state.challengeJudgeModels.value.length > 3) {
    errors.push('Select one through three B/C verification judges.')
  }
  const challengeJudges = state.challengeJudgeModels.value.map((id) => model(models, id))
  if (challengeJudges.some((judge) => judge === undefined || !supports(judge, 'judge'))) {
    errors.push('Select only available B/C verification judge models.')
  }
  if (new Set(state.challengeJudgeModels.value).size !== state.challengeJudgeModels.value.length) {
    errors.push('B/C verification judge selections must be unique.')
  }
  const selectedEvaluator = model(models, state.proposalEvaluatorModel.value)
  if (!selectedEvaluator || !supports(selectedEvaluator, 'proposal_evaluator')) {
    errors.push('Select an available proposal evaluator.')
  }
  const selectedModels = [selectedWriter, ...judges, ...challengeJudges, selectedEvaluator]
    .filter((candidate): candidate is ModelDescriptor => candidate !== undefined)
  if (selectedModels.some((candidate) => !fits(candidate, models, sessions))) {
    errors.push('Select models that fit the selected sessions.')
  }
  const rubricIds = new Set(rubrics.rubrics.map((rubric) => rubric.id))
  if (!state.rubricIds.length) errors.push('Select at least one rubric.')
  else if (state.rubricIds.some((id) => !rubricIds.has(id)) ||
    new Set(state.rubricIds).size !== state.rubricIds.length) {
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
  sessions: readonly SessionTokens[] = [],
): RunConfig {
  const assessment = assessRunConfig(state, models, rubrics, sessions)
  if (assessment.errors.length) throw new Error(assessment.errors[0])
  return {
    model_catalog_version: models.catalog_version,
    rubric_catalog_version: rubrics.catalog_version,
    judge_models: [...state.judgeModels.value],
    challenge_judge_models: [...state.challengeJudgeModels.value],
    proposal_model: state.proposalModel.value,
    proposal_evaluator_model: state.proposalEvaluatorModel.value,
    rubrics: [...state.rubricIds],
    candidate_budget: state.candidateBudget,
    force: state.force,
  }
}
