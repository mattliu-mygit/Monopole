import type { PromotionAvailability } from './reflectionState'

export default function ReflectionDecisionControls({
  availability,
  hasSavedDraft,
  hasUnsavedChanges,
  actionMismatchPaths,
  busy,
  canDismiss,
  acknowledgedUnevaluated,
  acknowledgedUnverified,
  onAcknowledgedUnevaluatedChange,
  onAcknowledgedUnverifiedChange,
  onSaveDraft,
  onResetDraft,
  onPromote,
  onDismiss,
}: {
  availability: PromotionAvailability
  hasSavedDraft: boolean
  hasUnsavedChanges: boolean
  actionMismatchPaths: string[]
  busy: boolean
  canDismiss: boolean
  acknowledgedUnevaluated: boolean
  acknowledgedUnverified: boolean
  onAcknowledgedUnevaluatedChange: (value: boolean) => void
  onAcknowledgedUnverifiedChange: (value: boolean) => void
  onSaveDraft: () => void
  onResetDraft: () => void
  onPromote: () => void
  onDismiss: () => void
}) {
  const isEdited = availability.target === 'edited-candidate'
  const isUnverified = availability.requiresUnverifiedAcknowledgement
  const invalidActionSet = actionMismatchPaths.length > 0
  const promoteDisabled = !availability.enabled || busy || hasUnsavedChanges ||
    (isEdited && !acknowledgedUnevaluated) || (isUnverified && !acknowledgedUnverified)

  return (
    <section id="reflection-decision" aria-label="Reflection decision" className="space-y-3 rounded-lg border border-gray-200 p-4">
      {availability.reason && <p className="text-sm text-gray-600">{availability.reason}</p>}

      {isEdited && (
        <label className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <input
            type="checkbox"
            checked={acknowledgedUnevaluated}
            disabled={busy}
            onChange={(event) => onAcknowledgedUnevaluatedChange(event.target.checked)}
          />
          <span>I understand edited proposal C differs from evaluated B and has not been evaluated.</span>
        </label>
      )}

      {isUnverified && (
        <label className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <input
            type="checkbox"
            checked={acknowledgedUnverified}
            disabled={busy}
            onChange={(event) => onAcknowledgedUnverifiedChange(event.target.checked)}
          />
          <span>I understand proposal B did not pass paired sandbox verification.</span>
        </label>
      )}

      {availability.enabled && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          Files are applied one at a time. If a later file fails, earlier complete files remain applied and the receipt lists every outcome.
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        <button type="button" disabled={busy || !availability.enabled || !hasUnsavedChanges || invalidActionSet} onClick={onSaveDraft} className="rounded-lg border border-blue-300 bg-blue-50 px-3 py-2 text-sm font-medium text-blue-800 hover:bg-blue-100 disabled:opacity-50">
          Save edited C
        </button>
        {(hasUnsavedChanges || hasSavedDraft) && (
          <button type="button" disabled={busy || (!availability.enabled && hasSavedDraft)} onClick={onResetDraft} className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">
            Reset to evaluated B
          </button>
        )}
        <button type="button" disabled={promoteDisabled} onClick={onPromote} className="rounded-lg bg-green-700 px-3 py-2 text-sm font-semibold text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300">
          {isEdited ? 'Promote unevaluated C' : isUnverified ? 'Promote unverified B' : 'Promote evaluated B'}
        </button>
        <button type="button" disabled={!canDismiss || busy} onClick={onDismiss} className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">
          Dismiss proposal
        </button>
      </div>
      {hasUnsavedChanges && (
        <p className="text-xs text-amber-700">Save or reset your inline edits before promotion.</p>
      )}
      {invalidActionSet && (
        <p role="alert" className="text-xs text-red-700">
          C changes the evaluated action set for {actionMismatchPaths.join(', ')}. Reset to evaluated B before saving.
        </p>
      )}
    </section>
  )
}
