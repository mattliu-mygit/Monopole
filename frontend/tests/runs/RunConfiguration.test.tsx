// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ModelCatalog, ModelDescriptor, RubricCatalog } from '../../src/types'
import type { RunConfigState } from '../../src/features/runs/runConfigState'
import RunConfiguration from '../../src/features/runs/RunConfiguration'

afterEach(cleanup)

const writerOpenAI: ModelDescriptor = {
  id: 'writer-openai',
  label: 'Writer OpenAI',
  family: 'openai',
  backend: 'cli',
  supported_roles: ['proposal_writer'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const writerAnthropic: ModelDescriptor = {
  id: 'writer-anthropic',
  label: 'Writer Anthropic',
  family: 'anthropic',
  backend: 'cli',
  supported_roles: ['proposal_writer'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const cliAnthropic: ModelDescriptor = {
  id: 'cli-anthropic',
  label: 'CLI Anthropic',
  family: 'anthropic',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const cliOpenAI: ModelDescriptor = {
  id: 'cli-openai',
  label: 'CLI OpenAI',
  family: 'openai',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const cliMeta: ModelDescriptor = {
  id: 'cli-meta',
  label: 'CLI Meta',
  family: 'meta',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const cliMetaAlt: ModelDescriptor = {
  id: 'cli-meta-alt',
  label: 'CLI Meta Alternate',
  family: 'meta',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}

const models: ModelCatalog = {
  catalog_version: 'models-v1',
  proposal: {
    available_models: [writerOpenAI, writerAnthropic],
    recommended_model: writerOpenAI.id,
  },
  recommended_judge_backend: 'cli',
  judge_backends: {
    cli: {
      available_models: [cliAnthropic, cliOpenAI, cliMeta, cliMetaAlt],
      recommended_judges: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
      proposal_evaluator_preferences: [cliOpenAI.id, cliAnthropic.id, cliMeta.id],
    },
    wandb: {
      available_models: [],
      recommended_judges: [],
      proposal_evaluator_preferences: [],
    },
  },
}

const rubrics: RubricCatalog = {
  catalog_version: 'rubrics-v1',
  rubrics: [
    {
      id: 'judge.verification',
      label: 'Verification discipline',
      evaluation_unit: 'session',
      version: 'v1',
      content_digest: 'verification-digest',
      pass_threshold: 0.5,
    },
    {
      id: 'judge.session_outcome',
      label: 'Session outcome',
      evaluation_unit: 'session',
      version: 'v2',
      content_digest: 'session-digest',
      pass_threshold: 0.7,
    },
  ],
}

const state: RunConfigState = {
  proposalModel: { value: writerOpenAI.id, source: 'recommended' },
  judgeBackend: { value: 'cli', source: 'recommended' },
  judgeModels: {
    value: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
    source: 'recommended',
  },
  proposalEvaluatorModel: { value: cliAnthropic.id, source: 'automatic' },
  rubricIds: ['judge.verification', 'judge.session_outcome'],
  candidateBudget: 3,
  force: false,
}

describe('RunConfiguration', () => {
  it('renders every model role as a guided select with visible choice provenance', () => {
    render(
      <RunConfiguration
        state={state}
        models={models}
        rubrics={rubrics}
        onAction={() => undefined}
      />,
    )

    expect(screen.getByRole('combobox', { name: 'Proposal writer' })).not.toBeNull()
    expect(screen.getByRole('combobox', { name: 'Judge backend' })).not.toBeNull()
    expect(screen.getByRole('combobox', { name: 'Judge 1' })).not.toBeNull()
    expect(screen.getByRole('combobox', { name: 'Judge 2' })).not.toBeNull()
    expect(screen.getByRole('combobox', { name: 'Judge 3' })).not.toBeNull()
    expect(screen.getByRole('combobox', { name: 'Proposal evaluator' })).not.toBeNull()
    expect(screen.getByLabelText('Proposal writer source').textContent).toBe('recommended')
    expect(screen.getByLabelText('Ordered judges source').textContent).toBe('recommended')
    expect(screen.getByLabelText('Proposal evaluator source').textContent).toBe('automatic')
    expect(screen.queryByRole('textbox')).toBeNull()
  })

  it('emits only reducer actions for controlled configuration changes', () => {
    const onAction = vi.fn()
    render(
      <RunConfiguration
        state={state}
        models={models}
        rubrics={rubrics}
        onAction={onAction}
      />,
    )

    fireEvent.change(screen.getByRole('combobox', { name: 'Proposal writer' }), {
      target: { value: writerAnthropic.id },
    })
    fireEvent.change(screen.getByRole('combobox', { name: 'Judge backend' }), {
      target: { value: 'wandb' },
    })
    fireEvent.change(screen.getByRole('combobox', { name: 'Judge 1' }), {
      target: { value: cliMeta.id },
    })
    fireEvent.change(screen.getByRole('combobox', { name: 'Proposal evaluator' }), {
      target: { value: cliMeta.id },
    })
    fireEvent.click(screen.getByRole('checkbox', { name: 'Verification discipline' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Proposal attempt limit' }), {
      target: { value: '5' },
    })
    expect(screen.getByRole('spinbutton', { name: 'Proposal attempt limit' }).getAttribute('max')).toBe('10')
    fireEvent.click(screen.getByRole('checkbox', { name: 'Force replacement scoring' }))
    fireEvent.click(screen.getByRole('button', { name: 'Use recommended configuration' }))

    expect(onAction.mock.calls.map(([action]) => action)).toEqual([
      { type: 'select-writer', modelId: writerAnthropic.id },
      { type: 'select-backend', backend: 'wandb' },
      { type: 'select-judge', position: 1, modelId: cliMeta.id },
      { type: 'select-evaluator', modelId: cliMeta.id },
      { type: 'set-rubrics', rubricIds: ['judge.session_outcome'] },
      { type: 'set-budget', value: 5 },
      { type: 'set-force', value: true },
      { type: 'use-recommended' },
    ])
  })

  it('supports one through three judges and exposes blocking errors separately from warnings', () => {
    const onAction = vi.fn()
    const twoJudgeState: RunConfigState = {
      ...state,
      judgeModels: {
        value: [cliAnthropic.id, cliOpenAI.id],
        source: 'overridden',
      },
    }
    const { rerender } = render(
      <RunConfiguration
        state={twoJudgeState}
        models={models}
        rubrics={rubrics}
        onAction={onAction}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Add judge' }))
    expect(onAction).toHaveBeenLastCalledWith({
      type: 'select-judge',
      position: 3,
      modelId: cliMeta.id,
    })

    const warningState: RunConfigState = {
      ...state,
      proposalModel: { value: 'missing-writer', source: 'overridden' },
      judgeModels: {
        value: [cliAnthropic.id, cliMeta.id, cliMetaAlt.id],
        source: 'overridden',
      },
    }
    rerender(
      <RunConfiguration
        state={warningState}
        models={models}
        rubrics={rubrics}
        onAction={onAction}
      />,
    )

    expect(screen.getByRole('alert').textContent).toContain(
      'Select an available proposal writer.',
    )
    expect(screen.getByRole('status').textContent).toContain(
      'Multiple selected judges share the meta model family.',
    )
    fireEvent.click(screen.getByRole('button', { name: 'Remove last judge' }))
    expect(onAction).toHaveBeenLastCalledWith({ type: 'remove-last-judge' })
  })

  it('keeps the selected rubrics explicit and restores the full catalog explicitly', () => {
    const onAction = vi.fn()
    render(
      <RunConfiguration
        state={{ ...state, rubricIds: ['judge.verification'] }}
        models={models}
        rubrics={rubrics}
        onAction={onAction}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Select all rubrics' }))

    expect(onAction).toHaveBeenCalledWith({
      type: 'set-rubrics',
      rubricIds: ['judge.verification', 'judge.session_outcome'],
    })
    expect(screen.queryByText(/stored as an empty list/i)).toBeNull()
  })

  it('labels the writer-call budget as the proposal attempt limit', () => {
    render(
      <RunConfiguration
        state={state}
        models={models}
        rubrics={rubrics}
        onAction={() => undefined}
      />,
    )

    expect(screen.getByRole('spinbutton', { name: 'Proposal attempt limit' })).not.toBeNull()
    expect(screen.queryByText('Candidate budget')).toBeNull()
  })
})
