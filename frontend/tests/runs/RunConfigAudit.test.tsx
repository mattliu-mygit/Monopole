// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { EffectiveRunConfig } from '../../src/types'
import RunConfigAudit from '../../src/features/runs/RunConfigAudit'

afterEach(cleanup)

const config: EffectiveRunConfig = {
  schema_version: '5',
  pipeline_version: '8',
  model_catalog_version: 'models-v4',
  rubric_catalog_version: 'rubrics-v9',
  models: {
    proposal_writer: {
      id: 'writer-openai',
      label: 'Writer OpenAI',
      family: 'openai',
      provider: 'codex',
      provider_model: 'writer-openai',
      supported_roles: ['proposal_writer'],
      max_input_tokens: 128_000,
      token_counter: 'utf8_bytes_div_3',
    },
    judges: [
      {
        id: 'judge-anthropic',
        label: 'Judge Anthropic',
        family: 'anthropic',
        provider: 'claude',
        provider_model: 'judge-anthropic',
        supported_roles: ['judge'],
        max_input_tokens: 128_000,
        token_counter: 'utf8_bytes_div_3',
        role: 'judge',
        position: 1,
      },
      {
        id: 'judge-meta',
        label: 'Judge Meta',
        family: 'meta',
        provider: 'agy',
        provider_model: 'judge-meta',
        supported_roles: ['judge', 'proposal_evaluator'],
        max_input_tokens: 128_000,
        token_counter: 'utf8_bytes_div_3',
        role: 'judge',
        position: 2,
      },
    ],
    challenge_judges: [
      {
        id: 'judge-meta',
        label: 'Judge Meta',
        family: 'meta',
        provider: 'agy',
        provider_model: 'judge-meta',
        supported_roles: ['judge', 'proposal_evaluator'],
        max_input_tokens: 128_000,
        token_counter: 'utf8_bytes_div_3',
        role: 'judge',
        position: 1,
      },
    ],
    proposal_evaluator: {
      id: 'judge-meta',
      label: 'Judge Meta',
      family: 'meta',
      provider: 'agy',
      provider_model: 'judge-meta',
      supported_roles: ['judge', 'proposal_evaluator'],
      max_input_tokens: 128_000,
      token_counter: 'utf8_bytes_div_3',
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
    contract_version: '3', large_model_threshold_tokens: 200_000,
    large_model_reserve_tokens: 100_000, small_model_reserve_tokens: 50_000,
    large_model_raw_target_tokens: 128_000, small_model_raw_target_tokens: 50_000,
    prompt_reserve_tokens: 6_000,
    output_reserve_tokens: 4_000, safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 4_000, overlap_turns: 1, max_chunks: 40,
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
    expect(screen.getAllByText('Judge Meta').length).toBe(3)
    expect(screen.getByText('Judge 1')).not.toBeNull()
    expect(screen.getByText('Judge 2')).not.toBeNull()
    expect(screen.getByText('Judge panel')).not.toBeNull()
    expect(screen.getByText('2 judges')).not.toBeNull()
    expect(screen.getByText('B/C verification judges')).not.toBeNull()
    expect(screen.getByText('B/C judge 1')).not.toBeNull()
    expect(screen.getByText('B/C judge panel')).not.toBeNull()
    expect(screen.getByText('1 judge')).not.toBeNull()
    expect(screen.getByText('Verification discipline')).not.toBeNull()
    expect(screen.getByText('v3 · whole session · threshold 0.65')).not.toBeNull()
    expect(screen.getByText('v2 · whole session · threshold 0.50')).not.toBeNull()
    expect(screen.getByText('8')).not.toBeNull()
    expect(screen.getByText('models-v4')).not.toBeNull()
    expect(screen.getByText('rubrics-v9')).not.toBeNull()
    expect(screen.getByText('50,000 small-model · 100,000 large-model reserve')).not.toBeNull()
    expect(screen.getByText('50,000 small-model · 128,000 large-model raw target')).not.toBeNull()
    expect(screen.getByText('1,000 digest · 4,000 findings · 1 turn overlap')).not.toBeNull()
    expect(screen.getAllByText('128,000 max input tokens').length).toBe(5)
    expect(screen.getByText('Proposal attempt limit')).not.toBeNull()
    expect(screen.getByText('4 attempts')).not.toBeNull()
    expect(screen.queryByText('Candidate budget')).toBeNull()
    expect(screen.getByText('Enabled')).not.toBeNull()
    expect(screen.getByText('The proposal writer and evaluator share a model family.')).not.toBeNull()
  })
})
