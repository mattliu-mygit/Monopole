import type {
  JudgeBackendCatalog,
  ModelCatalog,
  ModelDescriptor,
  ReviewDepth,
  RubricCatalog,
} from '../../types'
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

function modelLabel(model: ModelDescriptor): string {
  return `${model.label} · ${model.family}`
}

function depthLabel(depth: ReviewDepth): string {
  if (depth === 'primary') return 'Primary — one judge'
  if (depth === 'selective') return 'Selective — second opinion near threshold'
  return 'Full panel — three judges'
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

function selectedBackend(
  models: ModelCatalog,
  backendName: string,
): JudgeBackendCatalog | undefined {
  return models.judge_backends[backendName]
}

function uniqueJudgeChoices(backend: JudgeBackendCatalog | undefined): ModelDescriptor[] {
  return backend?.available_models.filter((model) => supports(model, 'judge')) ?? []
}

function evaluatorChoices(backend: JudgeBackendCatalog | undefined): ModelDescriptor[] {
  return backend?.available_models.filter((model) => supports(model, 'proposal_evaluator')) ?? []
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
  disabled = false,
  onAction,
}: RunConfigurationProps) {
  const assessment = assessRunConfig(state, models, rubrics)
  const backend = selectedBackend(models, state.judgeBackend.value)
  const judgeChoices = uniqueJudgeChoices(backend)
  const evaluatorModels = evaluatorChoices(backend)
  const catalogRubricIds = rubrics.rubrics.map((rubric) => rubric.id)
  const selectedRubrics = new Set(state.rubricIds)
  const allRubrics =
    state.rubricIds.length === catalogRubricIds.length &&
    catalogRubricIds.every((rubricId) => selectedRubrics.has(rubricId))
  const orderedJudgeIds = [
    ...(backend?.recommended_judges ?? []),
    ...judgeChoices.map((model) => model.id),
  ].filter((modelId, index, values) => values.indexOf(modelId) === index)
  const addThirdJudgeId = orderedJudgeIds.find(
    (modelId) => !state.judgeModels.value.includes(modelId),
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
                choices={models.proposal.available_models}
              />
              {models.proposal.available_models.map((model) => (
                <option key={model.id} value={model.id}>
                  {modelLabel(model)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <ChoiceLabel
              htmlFor="judge-backend"
              source={state.judgeBackend.source}
            >
              Judge backend
            </ChoiceLabel>
            <select
              id="judge-backend"
              value={state.judgeBackend.value}
              onChange={(event) =>
                onAction({ type: 'select-backend', backend: event.target.value })
              }
              className={inputClass}
            >
              {!backend && (
                <option value={state.judgeBackend.value}>
                  Unavailable · {state.judgeBackend.value}
                </option>
              )}
              {Object.keys(models.judge_backends).map((backendName) => (
                <option key={backendName} value={backendName}>
                  {backendName}
                </option>
              ))}
            </select>
          </div>

          <div>
            <ChoiceLabel
              htmlFor="review-depth"
              source={state.reviewDepth.source}
            >
              Review depth
            </ChoiceLabel>
            <select
              id="review-depth"
              value={state.reviewDepth.value}
              onChange={(event) =>
                onAction({
                  type: 'select-depth',
                  depth: event.target.value as ReviewDepth,
                })
              }
              className={inputClass}
            >
              {!backend?.supported_review_depths.includes(state.reviewDepth.value) && (
                <option value={state.reviewDepth.value}>
                  Unavailable · {depthLabel(state.reviewDepth.value)}
                </option>
              )}
              {backend?.supported_review_depths.map((depth) => (
                <option key={depth} value={depth}>
                  {depthLabel(depth)}
                </option>
              ))}
            </select>
          </div>

          {state.reviewDepth.value === 'selective' && (
            <div>
              <label
                htmlFor="second-opinion-margin"
                className="mb-1 block text-sm font-medium text-gray-700"
              >
                Second-opinion margin
              </label>
              <input
                id="second-opinion-margin"
                type="number"
                min={0}
                max={0.5}
                step={0.01}
                value={state.secondOpinionMargin ?? ''}
                onChange={(event) =>
                  onAction({ type: 'set-margin', value: Number(event.target.value) })
                }
                className={inputClass}
              />
              <p className="mt-1 text-xs text-gray-500">
                Request another opinion when the first score is this close to the threshold.
              </p>
            </div>
          )}

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
                  {judgeChoices.map((model) => (
                    <option key={model.id} value={model.id}>
                      {modelLabel(model)}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
          {state.reviewDepth.value === 'selective' && (
            <div className="mt-2">
              {state.judgeModels.value.length < 3 ? (
                <button
                  type="button"
                  disabled={!addThirdJudgeId}
                  onClick={() => {
                    if (addThirdJudgeId) {
                      onAction({
                        type: 'select-judge',
                        position: 3,
                        modelId: addThirdJudgeId,
                      })
                    }
                  }}
                  className="text-xs font-medium text-blue-700 hover:text-blue-900 disabled:text-gray-400"
                >
                  Add third judge
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => onAction({ type: 'remove-third-judge' })}
                  className="text-xs font-medium text-blue-700 hover:text-blue-900"
                >
                  Remove third judge
                </button>
              )}
            </div>
          )}
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
            {evaluatorModels.map((model) => (
              <option key={model.id} value={model.id}>
                {modelLabel(model)}
              </option>
            ))}
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
