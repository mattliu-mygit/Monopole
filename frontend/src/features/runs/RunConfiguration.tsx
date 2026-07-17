import type {
  ModelCatalog,
  ModelDescriptor,
  RubricCatalog,
  SessionSummary,
} from '../../types'
import {
  estimateModelCapacity,
  type ModelCapacityEstimate,
} from './modelCapacity'
import {
  assessRunConfig,
  type ChoiceSource,
  type RunConfigAction,
  type RunConfigState,
} from './runConfigState'

export interface RunConfigurationProps {
  state: RunConfigState
  models: ModelCatalog
  rubrics: RubricCatalog
  sessions?: readonly Pick<SessionSummary, 'total_tokens' | 'largest_turn_tokens'>[]
  disabled?: boolean
  onAction: (action: RunConfigAction) => void
}

const inputClass =
  'block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500'

const sourceClass: Record<ChoiceSource, string> = {
  recommended: 'bg-blue-50 text-blue-700',
  automatic: 'bg-indigo-50 text-indigo-700',
  overridden: 'bg-gray-100 text-gray-700',
}

function SourceBadge({
  label,
  source,
}: {
  label: string
  source: ChoiceSource
}) {
  return (
    <span
      aria-label={`${label} source`}
      className={`rounded-full px-2 py-0.5 text-[0.6875rem] font-medium ${sourceClass[source]}`}
    >
      {source}
    </span>
  )
}

function supports(model: ModelDescriptor, role: 'judge' | 'proposal_evaluator'): boolean {
  return model.supported_roles.includes(role)
}

function formatTokens(tokens: number): string {
  if (tokens >= 1_000_000) return `${Number((tokens / 1_000_000).toFixed(2))}m`
  return `${Math.round(tokens / 1_000)}k`
}

function modelLabel(
  model: ModelDescriptor,
  estimate: ModelCapacityEstimate,
): string {
  const capacity = `${formatTokens(model.max_input_tokens)} context`
  const fit = estimate.fits === null
    ? ''
    : estimate.fits
      ? ` · fits (~${formatTokens(estimate.estimatedRequestTokens ?? 0)} max)`
      : ` · does not fit (~${formatTokens(estimate.estimatedRequestTokens ?? 0)} needed)`
  return `${model.label} · ${model.provider} · ${model.family} · ${capacity}${fit}`
}

function ModelOptions({
  choices,
  capacities,
}: {
  choices: readonly ModelDescriptor[]
  capacities: ReadonlyMap<string, ModelCapacityEstimate>
}) {
  return choices.map((model) => {
    const estimate = capacities.get(model.id)!
    return (
      <option key={model.id} value={model.id} disabled={estimate.fits === false}>
        {modelLabel(model, estimate)}
      </option>
    )
  })
}

function ChoiceLabel({
  htmlFor,
  children,
  source,
}: {
  htmlFor: string
  children: string
  source: ChoiceSource
}) {
  return (
    <div className="mb-1 flex items-center justify-between gap-2">
      <label htmlFor={htmlFor} className="text-sm font-medium text-gray-700">
        {children}
      </label>
      <SourceBadge label={children} source={source} />
    </div>
  )
}

function writerChoices(models: ModelCatalog): ModelDescriptor[] {
  return models.available_models.filter((model) =>
    model.supported_roles.includes('proposal_writer'))
}

function uniqueJudgeChoices(models: ModelCatalog): ModelDescriptor[] {
  return models.available_models.filter((model) => supports(model, 'judge'))
}

function evaluatorChoices(models: ModelCatalog): ModelDescriptor[] {
  return models.available_models.filter((model) => supports(model, 'proposal_evaluator'))
}

function UnavailableOption({
  value,
  choices,
}: {
  value: string
  choices: readonly ModelDescriptor[]
}) {
  if (!value || choices.some((choice) => choice.id === value)) return null
  return <option value={value}>Unavailable · {value}</option>
}

export default function RunConfiguration({
  state,
  models,
  rubrics,
  sessions = [],
  disabled = false,
  onAction,
}: RunConfigurationProps) {
  const assessment = assessRunConfig(state, models, rubrics, sessions)
  const proposalModels = writerChoices(models)
  const judgeChoices = uniqueJudgeChoices(models)
  const evaluatorModels = evaluatorChoices(models)
  const capacities = new Map(models.available_models.map((model) => [
    model.id,
    estimateModelCapacity(model, sessions, models.judging_context),
  ]))
  const fits = (modelId: string) => capacities.get(modelId)?.fits !== false
  const catalogRubricIds = rubrics.rubrics.map((rubric) => rubric.id)
  const selectedRubrics = new Set(state.rubricIds)
  const allRubrics =
    state.rubricIds.length === catalogRubricIds.length &&
    catalogRubricIds.every((rubricId) => selectedRubrics.has(rubricId))
  const orderedJudgeIds = [
    ...models.recommended_judges,
    ...judgeChoices.map((model) => model.id),
  ].filter((modelId, index, values) => values.indexOf(modelId) === index)
  const addJudgeId = orderedJudgeIds.find(
    (modelId) => !state.judgeModels.value.includes(modelId) && fits(modelId),
  )
  const orderedChallengeJudgeIds = [
    ...models.recommended_challenge_judges,
    ...models.recommended_judges,
    ...judgeChoices.map((model) => model.id),
  ].filter((modelId, index, values) => values.indexOf(modelId) === index)
  const addChallengeJudgeId = orderedChallengeJudgeIds.find(
    (modelId) => !state.challengeJudgeModels.value.includes(modelId) && fits(modelId),
  )

  function toggleRubric(rubricId: string) {
    const selected = new Set(state.rubricIds)
    if (selected.has(rubricId)) selected.delete(rubricId)
    else selected.add(rubricId)
    if (selected.size === 0) return
    onAction({
      type: 'set-rubrics',
      rubricIds: catalogRubricIds.filter((id) => selected.has(id)),
    })
  }

  return (
    <section aria-label="Run configuration" className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-gray-800">Pipeline configuration</h4>
          <p className="mt-0.5 text-xs text-gray-500">
            Every model is selected from the current versioned catalog.
          </p>
        </div>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onAction({ type: 'use-recommended' })}
          className="text-xs font-medium text-blue-700 hover:text-blue-900 disabled:text-gray-400"
        >
          Use recommended configuration
        </button>
      </div>

      <fieldset disabled={disabled} className="space-y-5">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <div>
            <ChoiceLabel
              htmlFor="proposal-writer"
              source={state.proposalModel.source}
            >
              Proposal writer
            </ChoiceLabel>
            <select
              id="proposal-writer"
              value={state.proposalModel.value}
              onChange={(event) =>
                onAction({ type: 'select-writer', modelId: event.target.value })
              }
              className={inputClass}
            >
              <UnavailableOption
                value={state.proposalModel.value}
                choices={proposalModels}
              />
              <ModelOptions choices={proposalModels} capacities={capacities} />
            </select>
          </div>

          <div>
            <label
              htmlFor="candidate-budget"
              className="mb-1 block text-sm font-medium text-gray-700"
            >
              Proposal attempt limit
            </label>
            <input
              id="candidate-budget"
              type="number"
              min={1}
              max={10}
              step={1}
              value={state.candidateBudget}
              onChange={(event) =>
                onAction({ type: 'set-budget', value: Number(event.target.value) })
              }
              className={inputClass}
            />
            <p className="mt-1 text-xs text-gray-500">
              Maximum proposal writer calls; reflection may stop early after no improvement.
            </p>
          </div>

          <label className="flex items-center gap-2 self-center text-sm text-gray-700">
            <input
              type="checkbox"
              checked={state.force}
              onChange={(event) =>
                onAction({ type: 'set-force', value: event.target.checked })
              }
            />
            Force replacement scoring
          </label>
        </div>

        <fieldset>
          <legend className="mb-2 flex w-full items-center justify-between gap-2 text-sm font-medium text-gray-700">
            <span>Ordered judges</span>
            <SourceBadge label="Ordered judges" source={state.judgeModels.source} />
          </legend>
          <div className="grid gap-3 md:grid-cols-3">
            {state.judgeModels.value.map((modelId, index) => (
              <div key={index}>
                <label
                  htmlFor={`judge-${index + 1}`}
                  className="mb-1 block text-xs font-medium text-gray-600"
                >
                  Judge {index + 1}
                </label>
                <select
                  id={`judge-${index + 1}`}
                  value={modelId}
                  onChange={(event) =>
                    onAction({
                      type: 'select-judge',
                      position: index + 1,
                      modelId: event.target.value,
                    })
                  }
                  className={inputClass}
                >
                  <UnavailableOption value={modelId} choices={judgeChoices} />
                  <ModelOptions choices={judgeChoices} capacities={capacities} />
                </select>
              </div>
            ))}
          </div>
          <div className="mt-2 flex gap-4">
              {state.judgeModels.value.length < 3 && (
                <button
                  type="button"
                  disabled={!addJudgeId}
                  onClick={() => {
                    if (addJudgeId) {
                      onAction({
                        type: 'select-judge',
                        position: state.judgeModels.value.length + 1,
                        modelId: addJudgeId,
                      })
                    }
                  }}
                  className="text-xs font-medium text-blue-700 hover:text-blue-900 disabled:text-gray-400"
                >
                  Add judge
                </button>
              )}
              {state.judgeModels.value.length > 1 && (
                <button
                  type="button"
                  onClick={() => onAction({ type: 'remove-last-judge' })}
                  className="text-xs font-medium text-blue-700 hover:text-blue-900"
                >
                  Remove last judge
                </button>
              )}
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-2 flex w-full items-center justify-between gap-2 text-sm font-medium text-gray-700">
            <span>B/C verification judges</span>
            <SourceBadge
              label="B/C verification judges"
              source={state.challengeJudgeModels.source}
            />
          </legend>
          <p className="mb-2 text-xs text-gray-500">
            Paired proposal verification defaults to one judge and is configured independently.
          </p>
          <div className="grid gap-3 md:grid-cols-3">
            {state.challengeJudgeModels.value.map((modelId, index) => (
              <div key={index}>
                <label
                  htmlFor={`challenge-judge-${index + 1}`}
                  className="mb-1 block text-xs font-medium text-gray-600"
                >
                  B/C verification judge {index + 1}
                </label>
                <select
                  id={`challenge-judge-${index + 1}`}
                  value={modelId}
                  onChange={(event) =>
                    onAction({
                      type: 'select-challenge-judge',
                      position: index + 1,
                      modelId: event.target.value,
                    })
                  }
                  className={inputClass}
                >
                  <UnavailableOption value={modelId} choices={judgeChoices} />
                  <ModelOptions choices={judgeChoices} capacities={capacities} />
                </select>
              </div>
            ))}
          </div>
          <div className="mt-2 flex gap-4">
            {state.challengeJudgeModels.value.length < 3 && (
              <button
                type="button"
                disabled={!addChallengeJudgeId}
                onClick={() => {
                  if (addChallengeJudgeId) {
                    onAction({
                      type: 'select-challenge-judge',
                      position: state.challengeJudgeModels.value.length + 1,
                      modelId: addChallengeJudgeId,
                    })
                  }
                }}
                className="text-xs font-medium text-blue-700 hover:text-blue-900 disabled:text-gray-400"
              >
                Add B/C verification judge
              </button>
            )}
            {state.challengeJudgeModels.value.length > 1 && (
              <button
                type="button"
                onClick={() => onAction({ type: 'remove-last-challenge-judge' })}
                className="text-xs font-medium text-blue-700 hover:text-blue-900"
              >
                Remove last B/C verification judge
              </button>
            )}
          </div>
        </fieldset>

        <div className="max-w-xl">
          <ChoiceLabel
            htmlFor="proposal-evaluator"
            source={state.proposalEvaluatorModel.source}
          >
            Proposal evaluator
          </ChoiceLabel>
          <select
            id="proposal-evaluator"
            value={state.proposalEvaluatorModel.value}
            onChange={(event) =>
              onAction({ type: 'select-evaluator', modelId: event.target.value })
            }
            className={inputClass}
          >
            <UnavailableOption
              value={state.proposalEvaluatorModel.value}
              choices={evaluatorModels}
            />
            <ModelOptions choices={evaluatorModels} capacities={capacities} />
          </select>
        </div>

        <fieldset>
          <legend className="sr-only">Rubrics</legend>
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-sm font-medium text-gray-700">Rubrics</span>
            <button
              type="button"
              disabled={allRubrics}
              onClick={() =>
                onAction({ type: 'set-rubrics', rubricIds: catalogRubricIds })
              }
              className="text-xs font-medium text-blue-700 hover:text-blue-900 disabled:text-gray-400"
            >
              Select all rubrics
            </button>
          </div>
          <p className="mb-2 text-xs text-gray-500">
            The saved run records each rubric explicitly.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {rubrics.rubrics.map((rubric) => (
              <label
                key={rubric.id}
                className="flex items-start gap-2 rounded border border-gray-100 p-2 text-sm text-gray-700"
              >
                <input
                  className="mt-0.5"
                  type="checkbox"
                  aria-label={rubric.label}
                  checked={selectedRubrics.has(rubric.id)}
                  disabled={
                    state.rubricIds.length === 1 &&
                    state.rubricIds[0] === rubric.id
                  }
                  onChange={() => toggleRubric(rubric.id)}
                />
                <span>
                  <span className="block font-medium">{rubric.label}</span>
                  <span className="block font-mono text-[0.6875rem] text-gray-500">
                    {rubric.id} · whole session
                  </span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      </fieldset>

      {assessment.errors.length > 0 && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <div className="font-medium">Fix before starting</div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {assessment.errors.map((error) => <li key={error}>{error}</li>)}
          </ul>
        </div>
      )}
      {assessment.warnings.length > 0 && (
        <div role="status" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <div className="font-medium">Selection warnings</div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {assessment.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        </div>
      )}
    </section>
  )
}
