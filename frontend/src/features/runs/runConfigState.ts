import type {
  JudgeBackendCatalog,
  ModelCatalog,
  ModelDescriptor,
  ReviewDepth,
  RubricCatalog,
  RunConfig,
} from '../../types'

export type ChoiceSource = 'recommended' | 'automatic' | 'overridden'

export interface Choice<T> {
  value: T
  source: ChoiceSource
}

export interface RunConfigState {
  proposalModel: Choice<string>
  judgeBackend: Choice<string>
  reviewDepth: Choice<ReviewDepth>
  judgeModels: Choice<string[]>
  proposalEvaluatorModel: Choice<string>
  secondOpinionMargin: number | null
  rubricIds: string[]
  candidateBudget: number
  force: boolean
}

export type RunConfigAction =
  | { type: 'use-recommended' }
  | { type: 'select-writer'; modelId: string }
  | { type: 'select-backend'; backend: string }
  | { type: 'select-depth'; depth: ReviewDepth }
  | { type: 'select-judge'; position: number; modelId: string }
  | { type: 'remove-third-judge' }
  | { type: 'select-evaluator'; modelId: string }
  | { type: 'set-margin'; value: number }
  | { type: 'set-rubrics'; rubricIds: string[] }
  | { type: 'set-budget'; value: number }
  | { type: 'set-force'; value: boolean }

function sameValues(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function supportsRole(model: ModelDescriptor, role: ModelDescriptor['supported_roles'][number]): boolean {
  return model.supported_roles.includes(role)
}

function backendFor(
  models: ModelCatalog,
  backendName: string,
): JudgeBackendCatalog | undefined {
  return models.judge_backends[backendName]
}

function backendModel(
  backend: JudgeBackendCatalog | undefined,
  modelId: string,
): ModelDescriptor | undefined {
  return backend?.available_models.find((model) => model.id === modelId)
}

function proposalWriter(models: ModelCatalog, modelId: string): ModelDescriptor | undefined {
  return models.proposal.available_models.find((model) => model.id === modelId)
}

function marginDefault(models: ModelCatalog): number {
  return models.review_defaults.second_opinion_margin
}

function automaticEvaluator(
  models: ModelCatalog,
  backendName: string,
  writerId: string,
): string {
  const backend = backendFor(models, backendName)
  if (!backend) return ''

  const writerFamily = proposalWriter(models, writerId)?.family
  const preferred = backend.proposal_evaluator_preferences
    .map((modelId) => backendModel(backend, modelId))
    .filter((model): model is ModelDescriptor =>
      model !== undefined && supportsRole(model, 'proposal_evaluator'),
    )
  const available = backend.available_models.filter((model) =>
    supportsRole(model, 'proposal_evaluator'),
  )
  const candidates = [...preferred]
  for (const model of available) {
    if (!candidates.some((candidate) => candidate.id === model.id)) candidates.push(model)
  }

  return (
    candidates.find((model) => writerFamily === undefined || model.family !== writerFamily) ??
    candidates[0]
  )?.id ?? ''
}

function compatibleJudgeIds(
  backend: JudgeBackendCatalog | undefined,
  modelIds: readonly string[],
): boolean {
  return modelIds.every((modelId) => {
    const model = backendModel(backend, modelId)
    return model !== undefined && supportsRole(model, 'judge')
  })
}

function hasValidJudgeCount(depth: ReviewDepth, count: number): boolean {
  if (depth === 'primary') return count === 1
  if (depth === 'selective') return count === 2 || count === 3
  return count === 3
}

function judgeCandidates(backend: JudgeBackendCatalog | undefined): string[] {
  if (!backend) return []
  const candidates = [...backend.recommended_judges]
  for (const model of backend.available_models) {
    if (supportsRole(model, 'judge') && !candidates.includes(model.id)) candidates.push(model.id)
  }
  return candidates
}

function normalizedJudges(
  current: readonly string[],
  backend: JudgeBackendCatalog | undefined,
  depth: ReviewDepth,
): string[] {
  const compatibleCurrent = current.filter((modelId, index) =>
    current.indexOf(modelId) === index && compatibleJudgeIds(backend, [modelId]),
  )
  const candidates = judgeCandidates(backend)
  const ordered = [...compatibleCurrent]
  for (const modelId of candidates) {
    if (!ordered.includes(modelId)) ordered.push(modelId)
  }

  if (depth === 'primary') return ordered.slice(0, 1)
  if (depth === 'full_panel') return ordered.slice(0, 3)

  if (compatibleCurrent.length >= 2 && compatibleCurrent.length <= 3) {
    return compatibleCurrent
  }
  const recommendedCount = backend?.recommended_judges.length
  const target = recommendedCount === 2 || recommendedCount === 3 ? recommendedCount : 2
  return ordered.slice(0, target)
}

function recommendedState(models: ModelCatalog, rubricIds: readonly string[]): RunConfigState {
  const backendName = models.recommended_judge_backend
  const backend = backendFor(models, backendName)
  const writerId = models.proposal.recommended_model ?? ''
  const depth = backend?.recommended_review_depth ?? 'primary'

  return {
    proposalModel: { value: writerId, source: 'recommended' },
    judgeBackend: { value: backendName, source: 'recommended' },
    reviewDepth: { value: depth, source: 'recommended' },
    judgeModels: {
      value: normalizedJudges(backend?.recommended_judges ?? [], backend, depth),
      source: 'recommended',
    },
    proposalEvaluatorModel: {
      value: automaticEvaluator(models, backendName, writerId),
      source: 'automatic',
    },
    secondOpinionMargin: depth === 'selective' ? marginDefault(models) : null,
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

  const backend = backendFor(models, saved.judge_backend)
  const automaticEvaluatorId = automaticEvaluator(
    models,
    saved.judge_backend,
    saved.proposal_model,
  )

  return {
    proposalModel: {
      value: saved.proposal_model,
      source:
        saved.proposal_model === models.proposal.recommended_model
          ? 'recommended'
          : 'overridden',
    },
    judgeBackend: {
      value: saved.judge_backend,
      source:
        saved.judge_backend === models.recommended_judge_backend
          ? 'recommended'
          : 'overridden',
    },
    reviewDepth: {
      value: saved.review_depth,
      source:
        saved.review_depth === backend?.recommended_review_depth
          ? 'recommended'
          : 'overridden',
    },
    judgeModels: {
      value: [...saved.judge_models],
      source: sameValues(saved.judge_models, backend?.recommended_judges ?? [])
        ? 'recommended'
        : 'overridden',
    },
    proposalEvaluatorModel: {
      value: saved.proposal_evaluator_model,
      source:
        saved.proposal_evaluator_model === automaticEvaluatorId
          ? 'automatic'
          : 'overridden',
    },
    secondOpinionMargin: saved.second_opinion_margin,
    rubricIds: saved.rubrics.length > 0
      ? [...saved.rubrics]
      : rubrics.rubrics.map((rubric) => rubric.id),
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
    case 'use-recommended':
      return {
        ...recommendedState(models, state.rubricIds),
        rubricIds: [...state.rubricIds],
        candidateBudget: state.candidateBudget,
        force: state.force,
      }

    case 'select-writer':
      return {
        ...state,
        proposalModel: { value: action.modelId, source: 'overridden' },
        proposalEvaluatorModel:
          state.proposalEvaluatorModel.source === 'overridden'
            ? state.proposalEvaluatorModel
            : {
                value: automaticEvaluator(models, state.judgeBackend.value, action.modelId),
                source: 'automatic',
              },
      }

    case 'select-backend': {
      const backend = backendFor(models, action.backend)
      const preserveDepth =
        state.reviewDepth.source === 'overridden' &&
        backend?.supported_review_depths.includes(state.reviewDepth.value)
      const depth = preserveDepth
        ? state.reviewDepth.value
        : backend?.recommended_review_depth ??
          backend?.supported_review_depths[0] ??
          state.reviewDepth.value
      const depthChoice = preserveDepth
        ? state.reviewDepth
        : { value: depth, source: 'recommended' as const }
      const keepJudges =
        state.judgeModels.source === 'overridden' &&
        hasValidJudgeCount(depth, state.judgeModels.value.length) &&
        compatibleJudgeIds(backend, state.judgeModels.value)
      const evaluator = backendModel(backend, state.proposalEvaluatorModel.value)
      const keepEvaluator =
        evaluator !== undefined && supportsRole(evaluator, 'proposal_evaluator')

      return {
        ...state,
        judgeBackend: { value: action.backend, source: 'overridden' },
        reviewDepth: depthChoice,
        judgeModels: keepJudges
          ? { ...state.judgeModels, value: [...state.judgeModels.value] }
          : {
              value: normalizedJudges(backend?.recommended_judges ?? [], backend, depth),
              source:
                backend && state.judgeModels.source !== 'overridden'
                  ? 'recommended'
                  : 'automatic',
            },
        proposalEvaluatorModel: keepEvaluator
          ? state.proposalEvaluatorModel
          : {
              value: automaticEvaluator(models, action.backend, state.proposalModel.value),
              source: 'automatic',
            },
        secondOpinionMargin:
          depth === 'selective'
            ? state.secondOpinionMargin ?? marginDefault(models)
            : null,
      }
    }

    case 'select-depth': {
      const backend = backendFor(models, state.judgeBackend.value)
      return {
        ...state,
        reviewDepth: { value: action.depth, source: 'overridden' },
        judgeModels: {
          value: normalizedJudges(state.judgeModels.value, backend, action.depth),
          source: 'automatic',
        },
        secondOpinionMargin:
          action.depth === 'selective'
            ? state.secondOpinionMargin ?? marginDefault(models)
            : null,
      }
    }

    case 'select-judge': {
      if (action.position < 1 || action.position > 3) return state
      const judgeModels = [...state.judgeModels.value]
      if (action.position > judgeModels.length + 1) return state
      judgeModels[action.position - 1] = action.modelId
      return {
        ...state,
        judgeModels: { value: judgeModels, source: 'overridden' },
      }
    }

    case 'remove-third-judge':
      if (state.reviewDepth.value !== 'selective' || state.judgeModels.value.length < 3) {
        return state
      }
      return {
        ...state,
        judgeModels: {
          value: state.judgeModels.value.slice(0, 2),
          source: 'overridden',
        },
      }

    case 'select-evaluator':
      return {
        ...state,
        proposalEvaluatorModel: { value: action.modelId, source: 'overridden' },
      }

    case 'set-margin':
      return {
        ...state,
        secondOpinionMargin:
          state.reviewDepth.value === 'selective' ? action.value : null,
      }

    case 'set-rubrics':
      return { ...state, rubricIds: [...action.rubricIds] }

    case 'set-budget':
      return { ...state, candidateBudget: action.value }

    case 'set-force':
      return { ...state, force: action.value }
  }
}

export function assessRunConfig(
  state: RunConfigState,
  models: ModelCatalog,
  rubrics: RubricCatalog,
): { errors: string[]; warnings: string[] } {
  const errors: string[] = []
  const warnings: string[] = []
  const writer = proposalWriter(models, state.proposalModel.value)
  if (!writer || !supportsRole(writer, 'proposal_writer')) {
    errors.push('Select an available proposal writer.')
  }

  const backend = backendFor(models, state.judgeBackend.value)
  if (!backend) errors.push('Select an available judge backend.')

  if (backend && !backend.supported_review_depths.includes(state.reviewDepth.value)) {
    errors.push('Select a review depth supported by the judge backend.')
  }

  const judgeCount = state.judgeModels.value.length
  if (state.reviewDepth.value === 'primary' && judgeCount !== 1) {
    errors.push('Primary review requires exactly 1 judge.')
  } else if (
    state.reviewDepth.value === 'selective' &&
    judgeCount !== 2 &&
    judgeCount !== 3
  ) {
    errors.push('Selective review requires exactly 2 or 3 judges.')
  } else if (state.reviewDepth.value === 'full_panel' && judgeCount !== 3) {
    errors.push('Full-panel review requires exactly 3 judges.')
  }

  const judges = state.judgeModels.value.map((modelId) => backendModel(backend, modelId))
  if (judges.some((judge) =>
    judge === undefined ||
    judge.backend !== state.judgeBackend.value ||
    !supportsRole(judge, 'judge'),
  )) {
    errors.push('Select only available judge models.')
  }
  if (new Set(state.judgeModels.value).size !== state.judgeModels.value.length) {
    errors.push('Judge selections must be unique.')
  }

  const evaluator = backendModel(backend, state.proposalEvaluatorModel.value)
  if (
    !evaluator ||
    evaluator.backend !== state.judgeBackend.value ||
    !supportsRole(evaluator, 'proposal_evaluator')
  ) {
    errors.push('Select an available proposal evaluator.')
  }

  if (state.reviewDepth.value === 'selective') {
    if (
      state.secondOpinionMargin === null ||
      !Number.isFinite(state.secondOpinionMargin) ||
      state.secondOpinionMargin < 0 ||
      state.secondOpinionMargin > 0.5
    ) {
      errors.push('Second-opinion margin must be between 0 and 0.5.')
    }
  } else if (state.secondOpinionMargin !== null) {
    errors.push('Second-opinion margin is only available for selective review.')
  }

  const rubricIds = new Set(rubrics.rubrics.map((rubric) => rubric.id))
  if (state.rubricIds.length === 0) {
    errors.push('Select at least one rubric.')
  } else if (
    state.rubricIds.some((rubricId) => !rubricIds.has(rubricId)) ||
    new Set(state.rubricIds).size !== state.rubricIds.length
  ) {
    errors.push('Select only available rubrics.')
  }

  if (!Number.isInteger(state.candidateBudget) || state.candidateBudget < 1) {
    errors.push('Proposal attempt limit must be at least 1.')
  } else if (state.candidateBudget > 10) {
    errors.push('Proposal attempt limit must be between 1 and 10.')
  }

  const availableJudges = judges.filter((judge): judge is ModelDescriptor => judge !== undefined)
  const familyCounts = new Map<string, number>()
  for (const judge of availableJudges) {
    if (judge.family !== 'unknown') {
      familyCounts.set(judge.family, (familyCounts.get(judge.family) ?? 0) + 1)
    }
  }
  for (const [family, count] of [...familyCounts].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    if (count > 1) {
      warnings.push(`Multiple selected judges share the ${family} model family.`)
    }
  }
  if (
    writer &&
    evaluator &&
    writer.family !== 'unknown' &&
    writer.family === evaluator.family
  ) {
    warnings.push(`The proposal writer and evaluator share the ${writer.family} model family.`)
  }

  return { errors, warnings }
}

export function toRunConfig(
  state: RunConfigState,
  models: ModelCatalog,
  rubrics: RubricCatalog,
): RunConfig {
  const assessment = assessRunConfig(state, models, rubrics)
  if (assessment.errors.length > 0) throw new Error(assessment.errors[0])

  return {
    model_catalog_version: models.catalog_version,
    rubric_catalog_version: rubrics.catalog_version,
    judge_backend: state.judgeBackend.value,
    review_depth: state.reviewDepth.value,
    judge_models: [...state.judgeModels.value],
    second_opinion_margin: state.secondOpinionMargin,
    proposal_model: state.proposalModel.value,
    proposal_evaluator_model: state.proposalEvaluatorModel.value,
    rubrics: [...state.rubricIds],
    candidate_budget: state.candidateBudget,
    force: state.force,
  }
}
