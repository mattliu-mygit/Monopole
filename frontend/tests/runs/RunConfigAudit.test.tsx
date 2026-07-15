// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { EffectiveRunConfig } from '../../src/types'
import RunConfigAudit from '../../src/features/runs/RunConfigAudit'

afterEach(cleanup)

const config: EffectiveRunConfig = {
  schema_version: '2',
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
      max_input_tokens: 128_000,
    },
    judges: [
      {
        id: 'judge-anthropic',
        label: 'Judge Anthropic',
        family: 'anthropic',
        backend: 'cli',
        supported_roles: ['judge'],
        max_input_tokens: 128_000,
        role: 'judge',
        position: 1,
      },
      {
        id: 'judge-meta',
        label: 'Judge Meta',
        family: 'meta',
        backend: 'cli',
        supported_roles: ['judge', 'proposal_evaluator'],
        max_input_tokens: 128_000,
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
      max_input_tokens: 128_000,
    },
  },
  rubrics: [
    {
      id: 'judge.verification',
      label: 'Verification discipline',
      evaluation_unit: 'session',
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
  judging_context: {
    contract_version: '1', target_input_tokens: 100_000, prompt_reserve_tokens: 6_000,
    output_reserve_tokens: 4_000, safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 750, overlap_turns: 1, max_chunks: 40, token_estimator: 'utf8_bytes_div_3',
  },
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
    expect(screen.getByText('v3 · whole session · threshold 0.65')).not.toBeNull()
    expect(screen.getByText('v2 · whole session · threshold 0.50')).not.toBeNull()
    expect(screen.getByText('pipeline-v7')).not.toBeNull()
    expect(screen.getByText('models-v4')).not.toBeNull()
    expect(screen.getByText('rubrics-v9')).not.toBeNull()
    expect(screen.getByText('100,000 target input tokens')).not.toBeNull()
    expect(screen.getByText('1,000 digest · 750 findings · 1 turn overlap')).not.toBeNull()
    expect(screen.getAllByText('128,000 max input tokens').length).toBe(4)
    expect(screen.getByText('Proposal attempt limit')).not.toBeNull()
    expect(screen.getByText('4 attempts')).not.toBeNull()
    expect(screen.queryByText('Candidate budget')).toBeNull()
    expect(screen.getByText('Enabled')).not.toBeNull()
    expect(screen.getByText('The proposal writer and evaluator share a model family.')).not.toBeNull()
  })
})
