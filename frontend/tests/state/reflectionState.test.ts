import { describe, expect, it } from 'vitest'
import type {
  ModelDescriptor,
  PromotionReceipt,
  ReflectionBundleSnapshot,
  ReflectionCandidate,
  Run,
  SuccessfulReflectionResult,
} from '../../src/types'
import {
  bundleActions,
  bundlesEqual,
  changedLocators,
  promotionAvailability,
  receiptBundles,
  reviewState,
  selectedReflectionCandidate,
} from '../../src/features/reflection/reflectionState'

const writer: ModelDescriptor = {
  id: 'writer-1',
  label: 'Writer 1',
  family: 'writer-family',
  provider: 'codex',
  provider_model: 'writer-1',
  supported_roles: ['proposal_writer'],
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}

function bundle(revision: string, contents: Record<string, string | null>): ReflectionBundleSnapshot {
  return {
    revision,
    scope: { kind: 'file', target_id: '/project', patterns: ['CLAUDE.md'] },
    targets: Object.entries(contents).map(([locator, content]) => ({
      kind: 'file',
      locator,
      display_name: locator.split('/').at(-1) ?? locator,
      path: locator,
      exists: content !== null,
      content,
      revision: `${revision}:${locator}`,
    })),
  }
}

function candidate(id: string, value: ReflectionBundleSnapshot, score: number): ReflectionCandidate {
  return {
    candidate_id: id,
    bundle: value,
    score,
    score_delta: score - 0.4,
    rationale: `Why ${id}`,
    generation_attempt_id: `attempt-${id}`,
    requested_writer: writer,
    resolved_writer_model: writer.id,
    resolved_writer_family: writer.family,
    resolved_writer_backend: writer.provider,
    evaluation: {
      evaluation_id: `evaluation-${id}`,
      target_revision: value.revision,
      requested_model: 'evaluator-1',
      requested_family: 'evaluator-family',
      requested_backend: 'cli',
      resolved_model: 'evaluator-1',
      resolved_family: 'evaluator-family',
      resolved_backend: 'cli',
      score,
      rationale: `Why ${id}`,
      usage: {},
    },
  }
}

const past = bundle('bundle-b', { 'CLAUDE.md': 'past', '.claude/commands/old.md': 'old' })
const proposed = bundle('bundle-c', { 'CLAUDE.md': 'proposed', '.claude/skills/new.md': 'new' })
const edited = bundle('bundle-d', { 'CLAUDE.md': 'edited', '.claude/skills/new.md': 'new' })
const candidates = [candidate('candidate-1', proposed, 0.8), candidate('candidate-2', edited, 0.7)]

const result: SuccessfulReflectionResult = {
  baseline: past,
  baseline_score: 0.4,
  baseline_evaluation: {
    evaluation_id: 'evaluation-baseline',
    target_revision: past.revision,
    requested_model: 'evaluator-1',
    requested_family: 'evaluator-family',
    requested_backend: 'cli',
    resolved_model: 'evaluator-1',
    resolved_family: 'evaluator-family',
    resolved_backend: 'cli',
    score: 0.4,
    rationale: 'Baseline evaluation.',
    usage: {},
  },
  candidates,
  generation_attempts: [],
  recommended_candidate_id: 'candidate-1',
  baseline_won: false,
  reason: null,
  score_basis: 'predicted_evaluator',
  provisional_candidate_id: 'candidate-one',
  challenge: null,
}

function run(overrides: Partial<Run> = {}): Run {
  return {
    run_id: 'run-state',
    status: 'complete',
    current_stage_succeeded: true,
    created_at: '2026-07-14T18:00:00Z',
    auto_run: false,
    data_selection: null,
    run_config: null,
    effective_config: null,
    turn_cohort: null,
    judging_plan: null,
    reflection_input: null,
    scoring_progress: null,
    scoring_result: null,
    judging_progress: null,
    judging_result: null,
    reflecting_progress: null,
    reflecting_result: result,
    reflection_review: null,
    reflection_review_revision: 0,
    error: null,
    ...overrides,
  }
}

describe('reflectionState', () => {
  it('requires the exact persisted selection once review state exists', () => {
    expect(selectedReflectionCandidate(run({
      reflection_review: { status: 'pending', selected_candidate_id: 'candidate-2', draft: null },
    }))?.candidate_id).toBe('candidate-2')
    expect(selectedReflectionCandidate(run({
      reflection_review: { status: 'pending', selected_candidate_id: 'missing', draft: null },
    }))).toBeNull()
    expect(selectedReflectionCandidate(run({
      reflecting_result: { ...result, recommended_candidate_id: null },
    }))?.candidate_id).toBe('candidate-1')
  })

  it('derives pending, stale, final, and no-change review states', () => {
    expect(reviewState(run({
      reflection_review: { status: 'pending', selected_candidate_id: 'candidate-1', draft: null },
    }))).toBe('review-needed')
    expect(reviewState(run({
      reflection_review: {
        status: 'pending', selected_candidate_id: 'candidate-1', draft: null, stale: true,
      },
    }))).toBe('stale')
    expect(reviewState(run({
      reflection_review: { status: 'promoted', selected_candidate_id: 'candidate-1', draft: null },
    }))).toBe('promoted')
    expect(reviewState(run({
      reflection_review: { status: 'dismissed', selected_candidate_id: 'candidate-1', draft: null },
    }))).toBe('dismissed')
    expect(reviewState(run({
      reflecting_result: { ...result, baseline_won: true, candidates: [] },
    }))).toBe('no-change')
  })

  it('compares exact bundle revisions and derives every B-to-C action', () => {
    expect(bundlesEqual(proposed, { ...proposed })).toBe(true)
    expect(bundlesEqual(proposed, edited)).toBe(false)
    expect(bundleActions(past, proposed).map((action) => [action.action, action.locator])).toEqual([
      ['create', '.claude/skills/new.md'],
      ['update', 'CLAUDE.md'],
    ])
    expect(changedLocators(past, proposed)).toEqual([
      '.claude/skills/new.md',
      'CLAUDE.md',
    ])
  })

  it('distinguishes promotable evaluated C from unevaluated D', () => {
    const evaluated = run({
      reflection_review: { status: 'pending', selected_candidate_id: 'candidate-1', draft: null },
    })
    expect(promotionAvailability(evaluated)).toEqual({
      enabled: true,
      target: 'evaluated-candidate',
      requiresUnevaluatedAcknowledgement: false,
      reason: null,
    })

    const draft = run({
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-1',
        draft: { candidate_id: 'candidate-1', revision: edited.revision, bundle: edited },
      },
    })
    expect(promotionAvailability(draft)).toEqual({
      enabled: true,
      target: 'edited-candidate',
      requiresUnevaluatedAcknowledgement: true,
      reason: null,
    })
  })

  it('blocks mutations while finalizing, stale, resolved, or already mutating', () => {
    const pending = { status: 'pending' as const, selected_candidate_id: 'candidate-1', draft: null }
    expect(promotionAvailability(run({ status: 'reflecting', reflection_review: pending })).reason)
      .toMatch(/finalizing/i)
    expect(promotionAvailability(run({ reflection_review: { ...pending, stale: true } })).reason)
      .toMatch(/changed/i)
    expect(promotionAvailability(run({
      reflection_review: { status: 'dismissed', selected_candidate_id: 'candidate-1', draft: null },
    })).reason).toMatch(/resolved/i)
    expect(promotionAvailability(run({ reflection_review: pending }), true).reason)
      .toMatch(/in progress/i)
  })

  it('maps receipt evidence without treating promoted D as evaluated C', () => {
    const receipt: PromotionReceipt = {
      promotion_id: 'promotion-1',
      run_id: 'run-state',
      candidate_id: 'candidate-1',
      past,
      evaluated_candidate: proposed,
      promoted: edited,
      review_revision: 2,
      outcomes: [],
      decided_at: '2026-07-14T18:30:00Z',
      promoted_was_evaluated: false,
      unevaluated_d_acknowledged: true,
    }
    expect(receiptBundles(receipt)).toEqual({ past, evaluated: proposed, promoted: edited })
  })
})
