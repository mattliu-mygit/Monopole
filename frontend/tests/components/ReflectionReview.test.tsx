// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type {
  ModelDescriptor,
  PromotionReceipt,
  ReflectionBundleSnapshot,
  ReflectionReviewState,
  Run,
  SuccessfulReflectionResult,
} from '../../src/types'
import ReflectionReview from '../../src/features/reflection/ReflectionReview'

afterEach(cleanup)

const writer: ModelDescriptor = {
  id: 'writer-model',
  label: 'Writer',
  family: 'writer-family',
  backend: 'cli',
  supported_roles: ['proposal_writer'],
  max_input_tokens: 128_000,
}

function bundle(revision: string, claude: string): ReflectionBundleSnapshot {
  return {
    revision,
    scope: { kind: 'file', target_id: '/project', patterns: ['CLAUDE.md', '.claude/skills/*.md'] },
    targets: [
      {
        kind: 'file', locator: 'CLAUDE.md', display_name: 'CLAUDE.md', path: 'CLAUDE.md',
        exists: true, content: claude, revision: `${revision}:claude`,
      },
      {
        kind: 'file', locator: '.claude/skills/review.md', display_name: 'review.md', path: '.claude/skills/review.md',
        exists: revision !== 'b', content: revision === 'b' ? null : '# Review\nCheck the diff.', revision: `${revision}:review`,
      },
    ],
  }
}

const past = bundle('b', '# Past\nBe concise.')
const proposed = bundle('c', '# Proposed\nVerify before claiming success.')
const edited = bundle('d', '# Edited D\nVerify carefully.')

const result: SuccessfulReflectionResult = {
  baseline: past,
  baseline_score: 0.42,
  baseline_evaluation: {
    evaluation_id: 'evaluation-baseline',
    target_revision: past.revision,
    requested_model: 'evaluator-requested',
    requested_family: 'evaluator-family',
    requested_backend: 'cli',
    resolved_model: 'evaluator-resolved',
    resolved_family: 'evaluator-family',
    resolved_backend: 'cli',
    score: 0.42,
    rationale: 'Baseline evaluation.',
    usage: {},
  },
  candidates: [{
    candidate_id: 'candidate-one',
    bundle: proposed,
    score: 0.68,
    score_delta: 0.26,
    rationale: 'The weakest result was verification, so make the requirement explicit.',
    generation_attempt_id: 'attempt-1',
    requested_writer: writer,
    resolved_writer_model: 'writer-resolved',
    resolved_writer_family: 'writer-family',
    resolved_writer_backend: 'cli',
    evaluation: {
      evaluation_id: 'evaluation-1',
      target_revision: proposed.revision,
      requested_model: 'evaluator-requested',
      requested_family: 'evaluator-family',
      requested_backend: 'cli',
      resolved_model: 'evaluator-resolved',
      resolved_family: 'evaluator-family',
      resolved_backend: 'cli',
      score: 0.68,
      rationale: 'Better verification guidance.',
      usage: {},
    },
  }],
  generation_attempts: [],
  recommended_candidate_id: 'candidate-one',
  baseline_won: false,
  reason: null,
  score_basis: 'predicted_evaluator',
}

function multiCandidateResult(): SuccessfulReflectionResult {
  const alternative = bundle('c-two', '# Alternative C\nCover recovery paths.')
  return {
    ...result,
    candidates: [
      ...result.candidates,
      {
        ...result.candidates[0],
        candidate_id: 'candidate-two',
        bundle: alternative,
        score: 0.81,
        score_delta: 0.39,
        rationale: 'Alternative candidate rationale.',
        generation_attempt_id: 'attempt-2',
        evaluation: {
          ...result.candidates[0].evaluation,
          evaluation_id: 'evaluation-2',
          target_revision: alternative.revision,
          score: 0.81,
          rationale: 'Alternative evaluator rationale.',
        },
      },
    ],
  }
}

function run(overrides: Partial<Run> = {}): Run {
  return {
    run_id: 'run-review',
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
    reflection_review: {
      status: 'pending', selected_candidate_id: 'candidate-one', draft: null,
    },
    reflection_review_revision: 1,
    error: null,
    ...overrides,
  }
}

function receipt(): PromotionReceipt {
  return {
    promotion_id: 'promotion-1',
    run_id: 'run-review',
    candidate_id: 'candidate-one',
    target_kind: 'file',
    target_id: '/project',
    past,
    evaluated_candidate: proposed,
    promoted: edited,
    review_revision: 2,
    actions: [{
      action: 'update',
      locator: 'CLAUDE.md',
      before: past.targets[0],
      after: edited.targets[0],
    }],
    decided_at: '2026-07-14T18:30:00Z',
    promoted_was_evaluated: false,
    unevaluated_d_acknowledged: true,
    git_metadata: { branch: 'main' },
  }
}

describe('ReflectionReview', () => {
  it.each([
    'No evaluation feedback was found for the pinned cohort.',
    'No managed instruction targets were found.',
  ])('shows an early reflection exit without assuming B exists: %s', (reason) => {
    render(<ReflectionReview run={run({
      reflecting_result: { candidates: [], reason },
      reflection_review: null,
    })} />)

    expect(screen.getByText(reason)).not.toBeNull()
    expect(screen.getByText(/reflection ended before a baseline could be evaluated/i)).not.toBeNull()
    expect(screen.queryByText(/B evaluator audit/i)).toBeNull()
    expect(screen.queryByText(/Evaluated past B/i)).toBeNull()
  })

  it('leads with evaluated C versus B and shows evaluator provenance and every changed file', () => {
    render(<ReflectionReview run={run()} />)

    expect(screen.getAllByText('Past (B)').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Proposed (C)').length).toBeGreaterThan(0)
    expect(screen.getByText('0.420')).not.toBeNull()
    expect(screen.getByText('0.680')).not.toBeNull()
    expect(screen.getByText('+0.260')).not.toBeNull()
    expect(screen.getByText(/evaluated by evaluator-resolved/i)).not.toBeNull()
    expect(screen.getByRole('heading', { name: 'Evaluator assessment' })).not.toBeNull()
    expect(screen.getByRole('heading', { name: 'Writer provenance' })).not.toBeNull()
    expect(screen.getAllByText('Better verification guidance.').length).toBeGreaterThan(0)
    expect(screen.getByText(/predicted evaluator score, not a verification run/i)).not.toBeNull()
    fireEvent.click(screen.getByText('B evaluator audit'))
    expect(screen.getByText('evaluation-baseline')).not.toBeNull()
    expect(screen.getByText('Baseline evaluation.')).not.toBeNull()
    fireEvent.click(screen.getByText('C evaluator audit'))
    expect(screen.getByText('evaluation-1')).not.toBeNull()
    expect(screen.getByRole('button', { name: /CLAUDE\.md.*update/i })).not.toBeNull()
    expect(screen.getByRole('button', { name: /review\.md.*create/i })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /CLAUDE\.md.*update/i }))
    expect(screen.getByText(/\+ Verify before claiming success\./)).not.toBeNull()
  })

  it.each([
    ['without a review', null],
    ['after the review is resolved', {
      status: 'dismissed', selected_candidate_id: 'candidate-one', draft: null,
    } satisfies ReflectionReviewState],
  ])('locally inspects every evaluated candidate %s', (_label, review) => {
    render(<ReflectionReview run={run({
      reflecting_result: multiCandidateResult(),
      reflection_review: review,
    })} />)

    const alternative = screen.getByRole('button', {
      name: /Proposal 2.*attempt-2.*candidate-two/i,
    })
    expect(alternative.hasAttribute('disabled')).toBe(false)
    fireEvent.click(alternative)

    expect(alternative.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('0.810')).not.toBeNull()
    expect(screen.getAllByText('Alternative evaluator rationale.').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: /CLAUDE\.md.*update/i }))
    expect(screen.getByText((_, element) =>
      element?.tagName === 'PRE' && Boolean(element.textContent?.includes('+ Cover recovery paths.')),
    )).not.toBeNull()
  })

  it('keeps a pending server selection authoritative until the server returns a new run', async () => {
    const onSelect = vi.fn().mockResolvedValue(undefined)
    render(<ReflectionReview run={run({
      reflecting_result: multiCandidateResult(),
    })} onSelect={onSelect} />)

    fireEvent.click(screen.getByRole('button', {
      name: /Proposal 2.*attempt-2.*candidate-two/i,
    }))
    await waitFor(() => expect(onSelect).toHaveBeenCalledWith('candidate-two', false))

    expect(screen.getByRole('button', {
      name: /Proposal 1.*attempt-1.*candidate-one/i,
    }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.queryByText('0.810')).toBeNull()
  })

  it('associates bundle tabs with panels and supports arrow, Home, and End keys', () => {
    render(<ReflectionReview run={run()} />)

    const pastTab = screen.getByRole('tab', { name: 'Past (B)' })
    const proposedTab = screen.getByRole('tab', { name: 'Proposed (C)' })
    const diffTab = screen.getByRole('tab', { name: 'Diff B → C' })
    const controlledId = pastTab.getAttribute('aria-controls')
    expect(pastTab.id).not.toBe('')
    expect(controlledId).not.toBeNull()
    expect(document.getElementById(controlledId!)?.getAttribute('aria-labelledby')).toBe(pastTab.id)

    fireEvent.click(pastTab)
    fireEvent.keyDown(pastTab, { key: 'ArrowRight' })
    expect(proposedTab.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(proposedTab)

    fireEvent.keyDown(proposedTab, { key: 'End' })
    expect(diffTab.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(diffTab)

    fireEvent.keyDown(diffTab, { key: 'Home' })
    expect(pastTab.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(pastTab)
  })

  it('edits inline and sends only locator content for server-side hashing', async () => {
    const onSaveDraft = vi.fn().mockResolvedValue(undefined)
    render(<ReflectionReview run={run()} onSaveDraft={onSaveDraft} />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    expect(screen.getByRole('tab', { name: 'Edited (D)' }).getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(screen.getByLabelText('Edit CLAUDE.md'))
    fireEvent.click(screen.getByRole('button', { name: /CLAUDE\.md.*update/i }))
    fireEvent.click(screen.getByRole('tab', { name: 'Edited (D)' }))
    const editor = screen.getByLabelText('Edit CLAUDE.md')
    fireEvent.change(editor, { target: { value: '# Edited locally' } })
    fireEvent.click(screen.getByRole('tab', { name: 'Diff C → D' }))
    expect(screen.getByText((_, element) =>
      element?.tagName === 'PRE' && Boolean(element.textContent?.includes('- # Proposed')) &&
      Boolean(element.textContent?.includes('+ # Edited locally')),
    )).not.toBeNull()
    expect(screen.getByText(/D is not evaluated/i)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Save edited D' }))

    await waitFor(() => expect(onSaveDraft).toHaveBeenCalledWith({
      'CLAUDE.md': '# Edited locally',
      '.claude/skills/review.md': '# Review\nCheck the diff.',
    }))
  })

  it('keeps an unsaved D when equivalent run data is refreshed', () => {
    const onDirtyChange = vi.fn()
    const { rerender } = render(
      <ReflectionReview run={run()} onDirtyChange={onDirtyChange} />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved local D' },
    })

    rerender(
      <ReflectionReview
        run={run({ reflecting_result: structuredClone(result) })}
        onDirtyChange={onDirtyChange}
      />,
    )

    expect((screen.getByLabelText('Edit CLAUDE.md') as HTMLTextAreaElement).value)
      .toBe('# Unsaved local D')
    expect(onDirtyChange).toHaveBeenLastCalledWith(true)
  })

  it('warns that dismissal abandons unsaved D and clears it after dismissal succeeds', async () => {
    const onDismiss = vi.fn().mockResolvedValue(undefined)
    const onDirtyChange = vi.fn()
    render(
      <ReflectionReview
        run={run()}
        onDismiss={onDismiss}
        onDirtyChange={onDirtyChange}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved local D' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss proposal' }))

    expect(screen.getByRole('dialog', { name: 'Dismiss this proposal?' }).textContent)
      .toMatch(/discard your unsaved edited D/i)
    fireEvent.click(screen.getByRole('button', { name: 'Confirm dismissal' }))

    await waitFor(() => expect(onDismiss).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.queryByLabelText('Edit CLAUDE.md')).toBeNull())
    expect(onDirtyChange).toHaveBeenLastCalledWith(false)
  })

  it('clears unsaved D when a pending review becomes resolved externally', () => {
    const onDirtyChange = vi.fn()
    const { rerender } = render(
      <ReflectionReview run={run()} onDirtyChange={onDirtyChange} />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved local D' },
    })
    rerender(
      <ReflectionReview
        run={run({
          reflection_review: {
            status: 'dismissed', selected_candidate_id: 'candidate-one', draft: null,
          },
          reflection_review_revision: 2,
        })}
        onDirtyChange={onDirtyChange}
      />,
    )

    expect(screen.queryByLabelText('Edit CLAUDE.md')).toBeNull()
    expect(screen.queryByRole('region', { name: 'Reflection decision' })).toBeNull()
    expect(screen.getByText(/dismissed without changing managed instructions/i)).not.toBeNull()
    expect(onDirtyChange).toHaveBeenLastCalledWith(false)
  })

  it('keeps a dismissed saved D visible as evidence without mutation controls', () => {
    render(<ReflectionReview run={run({
      reflection_review: {
        status: 'dismissed',
        selected_candidate_id: 'candidate-one',
        draft: { candidate_id: 'candidate-one', revision: edited.revision, bundle: edited },
      },
      reflection_review_revision: 2,
    })} />)

    expect(screen.getByRole('tab', { name: 'Edited (D)' })).not.toBeNull()
    expect(screen.queryByRole('region', { name: 'Reflection decision' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Reset to evaluated C' })).toBeNull()
  })

  it.each([
    ['stale', {
      status: 'complete' as const,
      review: {
        status: 'pending' as const,
        selected_candidate_id: 'candidate-one',
        draft: null,
        stale: true,
        changed_targets: ['CLAUDE.md'],
        stale_reason: 'CLAUDE.md changed after evaluation.',
        current: bundle('current', '# Current'),
      },
    }],
    ['finalizing', {
      status: 'reflecting' as const,
      review: {
        status: 'pending' as const,
        selected_candidate_id: 'candidate-one',
        draft: null,
      },
    }],
  ])('locally inspects alternate candidates while a pending review is %s', (_label, state) => {
    const onSelect = vi.fn()
    const reflection = multiCandidateResult()
    const { rerender } = render(
      <ReflectionReview
        run={run({
          status: state.status,
          reflecting_result: reflection,
          reflection_review: state.review,
        })}
        onSelect={onSelect}
      />,
    )

    const alternative = screen.getByRole('button', {
      name: /Proposal 2.*attempt-2.*candidate-two/i,
    })
    expect(alternative.hasAttribute('disabled')).toBe(false)
    fireEvent.click(alternative)

    expect(alternative.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('0.810')).not.toBeNull()
    expect(screen.queryByRole('button', { name: 'Edit proposal inline' })).toBeNull()
    expect(onSelect).not.toHaveBeenCalled()

    rerender(
      <ReflectionReview
        run={run({
          status: state.status,
          reflecting_result: structuredClone(reflection),
          reflection_review: structuredClone(state.review),
        })}
        onSelect={onSelect}
      />,
    )
    expect(alternative.getAttribute('aria-pressed')).toBe('true')
  })

  it('disables stale draft writes while preserving local unsaved D for explicit reset', () => {
    const onSaveDraft = vi.fn()
    const current = bundle('current', '# Current')
    const { rerender } = render(
      <ReflectionReview run={run()} onSaveDraft={onSaveDraft} />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved local D' },
    })
    rerender(<ReflectionReview run={run({
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: null,
        stale: true,
        changed_targets: ['CLAUDE.md'],
        stale_reason: 'CLAUDE.md changed after evaluation.',
        current,
      },
    })} onSaveDraft={onSaveDraft} />)

    expect(screen.getByRole('button', { name: 'Save edited D' }).hasAttribute('disabled'))
      .toBe(true)
    expect(screen.getByRole('button', { name: 'Reset to evaluated C' }).hasAttribute('disabled'))
      .toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Reset to evaluated C' }))
    expect(screen.queryByLabelText('Edit CLAUDE.md')).toBeNull()
    expect(onSaveDraft).not.toHaveBeenCalled()
  })

  it('keeps unsaved D attached to its selected proposal while inspecting stale alternatives', () => {
    const reflection = multiCandidateResult()
    const onSelect = vi.fn()
    const { rerender } = render(
      <ReflectionReview
        run={run({ reflecting_result: reflection })}
        onSelect={onSelect}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved selected D' },
    })
    rerender(
      <ReflectionReview
        run={run({
          reflecting_result: structuredClone(reflection),
          reflection_review: {
            status: 'pending',
            selected_candidate_id: 'candidate-one',
            draft: null,
            stale: true,
            changed_targets: ['CLAUDE.md'],
            stale_reason: 'CLAUDE.md changed after evaluation.',
            current: bundle('current', '# Current'),
          },
        })}
        onSelect={onSelect}
      />,
    )

    fireEvent.click(screen.getByRole('button', {
      name: /Proposal 2.*attempt-2.*candidate-two/i,
    }))
    expect(screen.queryByLabelText('Edit CLAUDE.md')).toBeNull()
    expect(screen.queryByRole('region', { name: 'Reflection decision' })).toBeNull()

    fireEvent.click(screen.getByRole('button', {
      name: /Proposal 1.*attempt-1.*candidate-one/i,
    }))
    expect((screen.getByLabelText('Edit CLAUDE.md') as HTMLTextAreaElement).value)
      .toBe('# Unsaved selected D')
    expect(screen.getByRole('region', { name: 'Reflection decision' })).not.toBeNull()
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('closes blocked mutation dialogs when the review becomes stale', () => {
    const reflection = multiCandidateResult()
    const onSelect = vi.fn()
    const pending = run({ reflecting_result: reflection })
    const stale = run({
      reflecting_result: structuredClone(reflection),
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: null,
        stale: true,
        changed_targets: ['CLAUDE.md'],
        stale_reason: 'CLAUDE.md changed after evaluation.',
        current: bundle('current', '# Current'),
      },
    })
    const { rerender } = render(
      <ReflectionReview run={pending} onSelect={onSelect} />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Promote evaluated C' }))
    expect(screen.getByRole('dialog', { name: 'Promote evaluated C?' })).not.toBeNull()
    rerender(<ReflectionReview run={stale} onSelect={onSelect} />)
    expect(screen.queryByRole('dialog', { name: 'Promote evaluated C?' })).toBeNull()

    rerender(<ReflectionReview run={pending} onSelect={onSelect} />)
    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Unsaved selected D' },
    })
    fireEvent.click(screen.getByRole('button', {
      name: /Proposal 2.*attempt-2.*candidate-two/i,
    }))
    expect(screen.getByRole('dialog', { name: 'Discard edited D?' })).not.toBeNull()
    rerender(<ReflectionReview run={stale} onSelect={onSelect} />)

    expect(screen.queryByRole('dialog', { name: 'Discard edited D?' })).toBeNull()
    expect((screen.getByLabelText('Edit CLAUDE.md') as HTMLTextAreaElement).value)
      .toBe('# Unsaved selected D')
  })

  it('disables stale reset when D is already persisted and reset would write to the server', () => {
    render(<ReflectionReview run={run({
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: { candidate_id: 'candidate-one', revision: edited.revision, bundle: edited },
        stale: true,
        changed_targets: ['CLAUDE.md'],
        stale_reason: 'CLAUDE.md changed after evaluation.',
        current: bundle('current', '# Current'),
      },
    })} />)

    expect(screen.getByRole('button', { name: 'Reset to evaluated C' }).hasAttribute('disabled'))
      .toBe(true)
    expect(screen.getByRole('button', { name: 'Promote unevaluated D' }).hasAttribute('disabled'))
      .toBe(true)
    expect(screen.getByRole('button', { name: 'Dismiss proposal' }).hasAttribute('disabled'))
      .toBe(false)
  })

  it('blocks saving when D changes the evaluated action set and points to reset', () => {
    render(<ReflectionReview run={run()} onSaveDraft={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: '# Past\nBe concise.' },
    })

    expect(screen.getByRole('button', { name: 'Save edited D' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByText(/D changes the evaluated action set.*CLAUDE\.md/i)).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Reset to evaluated C' })).not.toBeNull()
  })

  it('promotes evaluated C without an unevaluated acknowledgement', async () => {
    const onPromote = vi.fn().mockResolvedValue(undefined)
    render(<ReflectionReview run={run()} onPromote={onPromote} />)

    fireEvent.click(screen.getByRole('button', { name: 'Promote evaluated C' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm promotion' }))
    await waitFor(() => expect(onPromote).toHaveBeenCalledWith(false))
  })

  it('marks D unevaluated and requires acknowledgement before promotion', async () => {
    const onPromote = vi.fn().mockResolvedValue(undefined)
    const withDraft = run({
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: { candidate_id: 'candidate-one', revision: edited.revision, bundle: edited },
      },
    })
    render(<ReflectionReview run={withDraft} onPromote={onPromote} />)

    const promote = screen.getByRole('button', { name: 'Promote unevaluated D' })
    expect(promote.hasAttribute('disabled')).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: /differs from evaluated C/i }))
    expect(promote.hasAttribute('disabled')).toBe(false)
    fireEvent.click(promote)
    fireEvent.click(screen.getByRole('button', { name: 'Confirm promotion' }))
    await waitFor(() => expect(onPromote).toHaveBeenCalledWith(true))
  })

  it('shows finalizing evidence without editable or promotion controls', () => {
    render(<ReflectionReview run={run({ status: 'reflecting' })} />)
    expect(screen.getByRole('status').textContent).toMatch(/finalizing review evidence/i)
    expect(screen.queryByRole('button', { name: /Promote evaluated C/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /Edit proposal inline/i })).toBeNull()
  })

  it('keeps exact B, evaluated C, and promoted D visible in the receipt', () => {
    render(<ReflectionReview run={run({
      reflection_review: {
        status: 'promoted',
        selected_candidate_id: 'candidate-one',
        draft: { candidate_id: 'candidate-one', revision: edited.revision, bundle: edited },
        receipt: receipt(),
      },
      reflection_review_revision: 2,
    })} />)

    expect(screen.getByRole('region', { name: 'Promotion receipt' })).not.toBeNull()
    expect(screen.queryByText('Evaluated C was promoted.')).toBeNull()
    expect(screen.getByText('Unevaluated edited D was promoted.')).not.toBeNull()
    expect(screen.getAllByText('Past B').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Evaluated C').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Promoted D').length).toBeGreaterThan(0)
    expect(screen.getByText('promotion-1')).not.toBeNull()
    expect(screen.getAllByText('candidate-one').length).toBeGreaterThan(0)
    expect(screen.getByText('/project')).not.toBeNull()
    expect(screen.getByText(/unevaluated D acknowledgement recorded/i)).not.toBeNull()
    expect(screen.getAllByText(/Diff C → D/i).length).toBeGreaterThan(0)
  })

  it('blocks stale review mutations and names changed managed files', () => {
    render(<ReflectionReview run={run({
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: null,
        stale: true,
        changed_targets: ['CLAUDE.md'],
        stale_reason: 'CLAUDE.md changed from the evaluated B revision.',
        current: bundle('current', '# Current\nChanged independently.'),
      },
    })} />)
    expect(screen.getByText('CLAUDE.md changed from the evaluated B revision.')).not.toBeNull()
    expect(screen.getAllByText(/Diff B → Current/).length).toBeGreaterThan(0)
    expect(screen.getByText((_, element) =>
      element?.tagName === 'PRE' && Boolean(element.textContent?.includes('+ # Current')),
    )).not.toBeNull()
    expect(screen.getByRole('link', { name: 'Start a new run' }).getAttribute('href')).toBe('/runs')
    expect(screen.getByRole('button', { name: /Promote evaluated C/i }).hasAttribute('disabled')).toBe(true)
    expect(screen.queryByRole('button', { name: 'Edit proposal inline' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Dismiss proposal' }).hasAttribute('disabled')).toBe(false)
  })

  it('distinguishes all-invalid reflection from an evaluated baseline winner', () => {
    const longExcerpt = `${'x'.repeat(2500)}TAIL-MUST-NOT-RENDER`
    render(<ReflectionReview run={run({
      reflecting_result: {
        ...result,
        candidates: [],
        generation_attempts: [{
          attempt_id: 'attempt-1',
          number: 1,
          status: 'failed',
          requested_writer: writer,
          resolved_model: 'writer-resolved',
          resolved_family: 'writer-family',
          resolved_backend: 'cli',
          usage: {},
          candidate_revision: null,
          changed_paths: ['../secret.md'],
          response_digest: `sha256:${'a'.repeat(64)}`,
          response_excerpt: longExcerpt,
          error_type: 'BundleValidationError',
          error: 'outside managed scope',
        }],
        recommended_candidate_id: null,
        baseline_won: false,
        reason: 'No valid proposal generated',
      },
      reflection_review: null,
    })} />)

    expect(screen.getByText('No valid proposal generated')).not.toBeNull()
    expect(screen.getByText('0.420')).not.toBeNull()
    expect(screen.getByText(/No proposed C was evaluated/i)).not.toBeNull()
    expect(screen.getAllByText('Baseline evaluation.').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByText('CLAUDE.md'))
    expect(screen.getByText((_, element) =>
      element?.tagName === 'PRE' && element.textContent === '# Past\nBe concise.',
    )).not.toBeNull()
    expect(screen.queryByText(/evaluated past B scored best/i)).toBeNull()
    expect(screen.queryByText(/baseline beat/i)).toBeNull()
    expect(screen.queryByRole('button', { name: /Promote/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /Edit proposal/i })).toBeNull()

    fireEvent.click(screen.getByText('Rejected proposal attempts'))
    expect(screen.getByText(/Attempt 1/)).not.toBeNull()
    expect(screen.getByText('attempt-1')).not.toBeNull()
    expect(screen.getByText('../secret.md')).not.toBeNull()
    expect(screen.getByText('BundleValidationError')).not.toBeNull()
    expect(screen.getByText('outside managed scope')).not.toBeNull()
    expect(screen.getByText(`sha256:${'a'.repeat(64)}`)).not.toBeNull()
    const excerpt = screen.getByLabelText('Attempt 1 response excerpt')
    expect(excerpt.textContent?.length).toBeLessThan(longExcerpt.length)
    expect(excerpt.textContent).not.toContain('TAIL-MUST-NOT-RENDER')
  })

  it('loads large baseline snapshots in bounded pages and mounts content only when opened', () => {
    const targets = Array.from({ length: 500 }, (_, index) => ({
      kind: 'file',
      locator: `instructions/file-${String(index).padStart(3, '0')}.md`,
      display_name: `file-${index}.md`,
      path: `instructions/file-${String(index).padStart(3, '0')}.md`,
      exists: true,
      content: `private-content-${index}`,
      revision: `b:${index}`,
    }))
    const largeBaseline = { ...past, revision: 'b-large', targets }
    render(<ReflectionReview run={run({
      reflecting_result: {
        ...result,
        baseline: largeBaseline,
        baseline_evaluation: {
          ...result.baseline_evaluation,
          target_revision: largeBaseline.revision,
        },
        candidates: [],
        recommended_candidate_id: null,
        baseline_won: true,
      },
      reflection_review: null,
    })} />)

    expect(screen.getByText('instructions/file-024.md')).not.toBeNull()
    expect(screen.queryByText('instructions/file-025.md')).toBeNull()
    expect(screen.queryByText('private-content-0')).toBeNull()
    expect(screen.queryByText('private-content-499')).toBeNull()

    fireEvent.click(screen.getByText('instructions/file-000.md'))
    expect(screen.getByText('private-content-0')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /show 25 more files/i }))
    expect(screen.getByText('instructions/file-049.md')).not.toBeNull()
    expect(screen.queryByText('instructions/file-050.md')).toBeNull()
    expect(screen.queryByText('private-content-49')).toBeNull()
  })

  it('does not substitute a different candidate when persisted evidence is missing', () => {
    render(<ReflectionReview run={run({
      reflection_review: {
        status: 'pending', selected_candidate_id: 'missing-candidate', draft: null,
      },
    })} />)

    expect(screen.getByText('Selected candidate evidence unavailable')).not.toBeNull()
    expect(screen.queryByRole('button', { name: /Promote/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /Dismiss/i })).toBeNull()
  })

  it('reloads saved multi-file D, resets D views when removed, and confirms candidate switches', async () => {
    const alternative = bundle('c-two', '# Alternative C\nKeep checks focused.')
    const second = {
      ...result.candidates[0],
      candidate_id: 'candidate-two',
      bundle: alternative,
      generation_attempt_id: 'attempt-2',
      evaluation: {
        ...result.candidates[0].evaluation,
        evaluation_id: 'evaluation-2',
        target_revision: alternative.revision,
      },
    }
    const multiResult = { ...result, candidates: [...result.candidates, second] }
    const saved = run({
      reflecting_result: multiResult,
      reflection_review: {
        status: 'pending',
        selected_candidate_id: 'candidate-one',
        draft: { candidate_id: 'candidate-one', revision: edited.revision, bundle: edited },
      },
    })
    const onSelect = vi.fn().mockResolvedValue(undefined)
    const { rerender } = render(<ReflectionReview run={saved} onSelect={onSelect} />)

    expect(screen.getByRole('tab', { name: 'Edited (D)' })).not.toBeNull()
    expect(screen.getByRole('tab', { name: 'Diff C → D' })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Proposal 2.*attempt-2.*candidate-two/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Discard D and select proposal' }))
    await waitFor(() => expect(onSelect).toHaveBeenCalledWith('candidate-two', true))

    rerender(<ReflectionReview run={run({
      reflecting_result: multiResult,
      reflection_review: {
        status: 'pending', selected_candidate_id: 'candidate-one', draft: null,
      },
      reflection_review_revision: 2,
    })} onSelect={onSelect} />)
    expect(screen.queryByRole('tab', { name: 'Edited (D)' })).toBeNull()
    expect(screen.getByRole('tab', { name: 'Diff B → C' }).getAttribute('aria-selected')).toBe('true')
  })
})
