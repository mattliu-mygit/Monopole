import { estimateModelCapacity, type CapacityRole } from "../../frontend/src/features/runs/modelCapacity";
import { assessRunConfig } from "../../frontend/src/features/runs/runConfigState";
import type { Catalog, ModelDescriptor, SessionEstimates } from "./protocol";

/** What New run has chosen. */
export interface Chosen {
  judges: string[];
  challengeJudges: string[];
  writer: string;
  evaluator: string;
  rubrics: string[];
  candidateBudget: number;
  force: boolean;
}

function tokens(count: number): string {
  return count >= 1_000_000 ? `${Number((count / 1_000_000).toFixed(2))}m` : `${Math.round(count / 1_000)}k`;
}

const ROLE_WORDS: Record<CapacityRole, string> = {
  judge: "judge",
  challenge_judge: "A/B check judge",
  proposal_writer: "proposal writer",
  proposal_evaluator: "evaluator",
};

/** The selected sessions' estimates in the shape the web app's capacity check reads; a session the
 * server has no estimate for is left out. */
export function capacitySessions(estimates: SessionEstimates | null, ids: string[]) {
  return ids.flatMap((id) => (estimates?.[id] ? [{ judging_token_estimates: estimates[id] }] : []));
}

/** Whether `model` fits the sessions in `role`, as the web app's option labels say: "fits", "does
 * not fit", or "" before any estimate. */
export function fitWords(catalog: Catalog, model: ModelDescriptor, role: CapacityRole, sessions: ReturnType<typeof capacitySessions>): string {
  const fits = estimateModelCapacity(model, sessions, catalog.models.judging_context, role).fits;
  return fits === null ? "" : fits ? "fits" : "does not fit";
}

/**
 * The web app's hints for New run's choices: each chosen model that doesn't fit the selected
 * sessions, with its estimated largest request and its limit, from the web app's capacity check
 * (frontend/src/features/runs/modelCapacity.ts), then its model-family diversity warnings
 * (assessRunConfig in runConfigState.ts). The server still decides; these only warn.
 */
export function runHints(catalog: Catalog, chosen: Chosen, sessions: ReturnType<typeof capacitySessions>): string[] {
  const models = catalog.models;
  const byId = new Map(models.available_models.map((model) => [model.id, model]));
  const roles: [string, CapacityRole][] = [
    ...chosen.judges.map((id): [string, CapacityRole] => [id, "judge"]),
    ...chosen.challengeJudges.map((id): [string, CapacityRole] => [id, "challenge_judge"]),
    [chosen.writer, "proposal_writer"],
    [chosen.evaluator, "proposal_evaluator"],
  ];
  const capacity = roles.flatMap(([id, role]) => {
    const model = byId.get(id);
    if (!model) return [];
    const estimate = estimateModelCapacity(model, sessions, models.judging_context, role);
    if (estimate.fits !== false) return [];
    const request = estimate.estimatedRequestTokens === null ? "" : `estimated largest request ~${tokens(estimate.estimatedRequestTokens)}, `;
    return [`${model.label}, as ${ROLE_WORDS[role]}, does not fit the selected sessions (${request}model limit ${tokens(model.max_input_tokens)}).`];
  });
  const overridden = <T>(value: T) => ({ value, source: "overridden" as const });
  const { warnings } = assessRunConfig(
    {
      proposalModel: overridden(chosen.writer),
      judgeModels: overridden(chosen.judges),
      challengeJudgeModels: overridden(chosen.challengeJudges),
      proposalEvaluatorModel: overridden(chosen.evaluator),
      rubricIds: chosen.rubrics,
      candidateBudget: chosen.candidateBudget,
      force: chosen.force,
    },
    models,
    catalog.rubrics,
    sessions,
  );
  return [...new Set([...capacity, ...warnings])];
}
