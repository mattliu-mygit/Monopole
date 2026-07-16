import type { ReactNode } from 'react'
import type { EffectiveRunConfig, ModelDescriptor } from '../../types'

export interface RunConfigAuditProps {
  config: EffectiveRunConfig
}

function ModelCard({
  title,
  model,
}: {
  title: string
  model: ModelDescriptor
}) {
  return (
    <div className="rounded-lg border border-gray-200 p-3">
      <div className="text-xs font-medium uppercase tracking-wide text-gray-500">{title}</div>
      <div className="mt-1 font-medium text-gray-900">{model.label}</div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs text-gray-500">
        <span>{model.id}</span>
        <span>{model.family}</span>
        <span>{model.backend}</span>
      </div>
      <div className="mt-1 text-xs text-gray-500">{model.max_input_tokens.toLocaleString()} max input tokens</div>
    </div>
  )
}

function AuditItem({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</dt>
      <dd className="mt-1 text-sm text-gray-800">{children}</dd>
    </div>
  )
}

export default function RunConfigAudit({ config }: RunConfigAuditProps) {
  return (
    <section
      aria-label="Pinned run configuration"
      className="space-y-5 rounded-lg border border-gray-200 bg-gray-50/60 p-4"
    >
      <div>
        <h4 className="text-sm font-semibold text-gray-900">Pinned configuration</h4>
        <p className="mt-0.5 text-xs text-gray-500">
          This immutable snapshot is the exact configuration used by the run.
        </p>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <ModelCard title="Proposal writer" model={config.models.proposal_writer} />
        <ModelCard title="Proposal evaluator" model={config.models.proposal_evaluator} />
      </div>

      <div>
        <h5 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
          Ordered judges
        </h5>
        <ol className="grid gap-3 md:grid-cols-3">
          {config.models.judges.map((judge) => (
            <li key={judge.position} className="rounded-lg border border-gray-200 bg-white p-3">
              <div className="text-xs font-medium text-gray-500">Judge {judge.position}</div>
              <div className="mt-1 font-medium text-gray-900">{judge.label}</div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs text-gray-500">
                <span>{judge.id}</span>
                <span>{judge.family}</span>
                <span>{judge.backend}</span>
              </div>
              <div className="mt-1 text-xs text-gray-500">{judge.max_input_tokens.toLocaleString()} max input tokens</div>
            </li>
          ))}
        </ol>
      </div>

      <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <AuditItem label="Judge panel">
          {config.models.judges.length} judge{config.models.judges.length === 1 ? '' : 's'}
        </AuditItem>
        <AuditItem label="Proposal attempt limit">
          {config.candidate_budget} attempt{config.candidate_budget === 1 ? '' : 's'}
        </AuditItem>
        <AuditItem label="Force replacement">{config.force ? 'Enabled' : 'Disabled'}</AuditItem>
        <AuditItem label="Pipeline version">
          <code>{config.pipeline_version}</code>
        </AuditItem>
        <AuditItem label="Model catalog">
          <code>{config.model_catalog_version}</code>
        </AuditItem>
        <AuditItem label="Rubric catalog">
          <code>{config.rubric_catalog_version}</code>
        </AuditItem>
        <AuditItem label="Schema version">
          <code>{config.schema_version}</code>
        </AuditItem>
      </dl>

      <div>
        <h5 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
          Sliding judging context
        </h5>
        <div className="rounded-lg border border-gray-200 bg-white p-3 text-xs text-gray-600">
          <div className="font-medium text-gray-900">
            {config.judging_context.target_input_tokens.toLocaleString()} target input tokens
          </div>
          <div className="mt-1">
            {config.judging_context.digest_max_tokens.toLocaleString()} digest ·{' '}
            {config.judging_context.finding_max_tokens.toLocaleString()} findings ·{' '}
            {config.judging_context.overlap_turns} turn overlap
          </div>
          <div className="mt-1">
            {config.judging_context.prompt_reserve_tokens.toLocaleString()} prompt ·{' '}
            {config.judging_context.output_reserve_tokens.toLocaleString()} output ·{' '}
            {config.judging_context.safety_reserve_tokens.toLocaleString()} safety reserve
          </div>
        </div>
      </div>

      <div>
        <h5 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
          Pinned rubrics
        </h5>
        <ul className="space-y-2">
          {config.rubrics.map((rubric) => (
            <li key={rubric.id} className="rounded-lg border border-gray-200 bg-white p-3">
              <div className="font-medium text-gray-900">{rubric.label}</div>
              <div className="mt-1 text-xs text-gray-500">
                {rubric.version} · whole session · threshold{' '}
                {rubric.pass_threshold.toFixed(2)}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[0.6875rem] text-gray-400">
                <span>{rubric.id}</span>
                <span>{rubric.content_digest}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>

      {config.selection_warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <h5 className="font-medium">Pinned selection warnings</h5>
          <ul className="mt-1 space-y-1">
            {config.selection_warnings.map((warning) => (
              <li key={warning.code}>
                <span>{warning.message}</span>{' '}
                <code className="text-xs text-amber-700">{warning.code}</code>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
