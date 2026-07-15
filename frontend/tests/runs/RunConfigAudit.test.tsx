// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { EffectiveRunConfig } from '../../src/types'
import RunConfigAudit from '../../src/features/runs/RunConfigAudit'

afterEach(cleanup)

const config: EffectiveRunConfig = {
  schema_version: '1',
  pipeline_version: 'pipeline-v7',
  model_catalog_version: 'models-v4',
  rubric_catalog_version: 'rubrics-v9',
  judge_backend: 'cli',
  review_depth: 'selective',
  second_opinion_margin: 0.12,
  models: {
    proposal_writer: {
      id: 'writer-openai',
      label: 'Writer OpenAI',
      family: 'openai',
      backend: 'cli',
      supported_roles: ['proposal_writer'],
    },
    judges: [
      {
        id: 'judge-anthropic',
        label: 'Judge Anthropic',
        family: 'anthropic',
        backend: 'cli',
        supported_roles: ['judge'],
        role: 'judge',
        position: 1,
      },
      {
        id: 'judge-meta',
        label: 'Judge Meta',
        family: 'meta',
        backend: 'cli',
        supported_roles: ['judge', 'proposal_evaluator'],
        role: 'judge',
        position: 2,
      },
    ],
    proposal_evaluator: {
      id: 'judge-meta',
      label: 'Judge Meta',
      family: 'meta',
      backend: 'cli',
      supported_roles: ['judge', 'proposal_evaluator'],
    },
  },
  rubrics: [
    {
      id: 'judge.verification',
      label: 'Verification discipline',
      evaluation_unit: 'episode',
      version: 'v3',
      content_digest: 'sha256:verification',
      pass_threshold: 0.65,
    },
    {
      id: 'judge.session_outcome',
      label: 'Session outcome',
      evaluation_unit: 'session',
      version: 'v2',
      content_digest: 'sha256:session',
      pass_threshold: 0.5,
    },
  ],
  selection_warnings: [
    {
      code: 'proposal_evaluator_writer_family_overlap',
      message: 'The proposal writer and evaluator share a model family.',
      affected_roles: ['proposal_writer', 'proposal_evaluator'],
      selected_model_ids: ['writer-openai', 'judge-meta'],
      compared_families: ['openai'],
    },
  ],
  candidate_budget: 4,
  force: true,
}

describe('RunConfigAudit', () => {
  it('shows the complete immutable effective configuration without a live catalog', () => {
    render(<RunConfigAudit config={config} />)

    expect(screen.getByRole('region', { name: 'Pinned run configuration' })).not.toBeNull()
    expect(screen.getByText('Writer OpenAI')).not.toBeNull()
    expect(screen.getByText('Judge Anthropic')).not.toBeNull()
    expect(screen.getAllByText('Judge Meta').length).toBe(2)
    expect(screen.getByText('Judge 1')).not.toBeNull()
    expect(screen.getByText('Judge 2')).not.toBeNull()
    expect(screen.getByText('Selective')).not.toBeNull()
    expect(screen.getByText('0.12')).not.toBeNull()
    expect(screen.getByText('Verification discipline')).not.toBeNull()
    expect(screen.getByText('v3 · episode · threshold 0.65')).not.toBeNull()
    expect(screen.getByText('v2 · whole session · threshold 0.50')).not.toBeNull()
    expect(screen.getByText('pipeline-v7')).not.toBeNull()
    expect(screen.getByText('models-v4')).not.toBeNull()
    expect(screen.getByText('rubrics-v9')).not.toBeNull()
    expect(screen.getByText('Proposal attempt limit')).not.toBeNull()
    expect(screen.getByText('4 attempts')).not.toBeNull()
    expect(screen.queryByText('Candidate budget')).toBeNull()
    expect(screen.getByText('Enabled')).not.toBeNull()
    expect(screen.getByText('The proposal writer and evaluator share a model family.')).not.toBeNull()
  })
})
