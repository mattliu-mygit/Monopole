import type { Card, Catalog, ProviderCheck, RunDetail, SessionList } from "./protocol";

// Invented runs in the shapes `bin/monopole runs` and `run` print, for development and tests.

const judges = [
  { id: "wandb:DeepSeek-V4.1-Flash", label: "DeepSeek V4.1 Flash", provider: "wandb" },
  { id: "codex:gpt-5.6-sol", label: "GPT-5.6 Sol", provider: "codex" },
];

const base = {
  session_count: 1,
  turns: 114,
  judges,
  rated: 0,
  mean_score: null,
  progress: null,
  failure: null,
  review_state: "none",
  proposal: null,
  deletable: true,
  next_stage: null,
} satisfies Partial<Card>;

export function cards(): Card[] {
  return [
    {
      ...base,
      run_id: "run-review01",
      status: "complete",
      label: "Review needed",
      dot: "attention",
      group: "review",
      created_at: "2026-11-26T16:14:29+00:00",
      updated_at: "2026-11-26T16:22:18+00:00",
      title: "Lets pause after scoring editing the skill copies",
      rated: 6,
      mean_score: 0.7083,
      review_state: "review-needed",
      proposal: {
        current: 0.15,
        proposed: 0.35,
        change: 0.2,
        candidates: 3,
        check: { status: "incomplete", winner: "tie", reason: "Sandbox preflight failed" },
        recommended: false,
        note: "Targets outcome.command and outcome.test.",
      },
    },
    {
      ...base,
      run_id: "run-failed01",
      status: "failed",
      label: "Failed at judging",
      dot: "error",
      group: "failed",
      created_at: "2026-11-05T22:00:56+00:00",
      updated_at: "2026-11-05T22:14:06+00:00",
      title: "Session 01a1ad64-7d0",
      turns: 351,
      failure: {
        stage: "judging",
        stopped_at: "Session Outcome Quality",
        cause: "Session Outcome Quality failed: DeepSeek V4.1 Flash failed ValueError after 28 provider requests.",
      },
    },
    {
      ...base,
      run_id: "run-running1",
      status: "judging",
      label: "Judging",
      dot: "running",
      group: "running",
      created_at: "2026-11-26T17:00:00+00:00",
      updated_at: "2026-11-26T17:05:00+00:00",
      title: "Fix the flaky login test",
      progress: "Judging 2 of 6 rubrics",
      deletable: false,
    },
    {
      ...base,
      run_id: "run-waiting1",
      status: "scoring",
      label: "Waiting to judge",
      dot: "attention",
      group: "waiting",
      created_at: "2026-11-26T16:00:00+00:00",
      updated_at: "2026-11-26T16:05:00+00:00",
      title: "Speed up the session list",
      progress: "Scoring complete: 12 scores written.",
      deletable: false,
      next_stage: "judging",
    },
  ];
}

export function detail(card: Card): RunDetail {
  return {
    ...card,
    selection: { session_ids: ["01a1ad64-7d00-75f7-b10f-e2e0471fdf58"], since: null, until: null, timezone: null },
    stages: [
      { stage: "scoring", state: "done", message: "Scoring complete: 0 scores written." },
      { stage: "judging", state: card.status === "failed" ? "failed" : "done", message: card.failure?.cause ?? "Judging complete" },
      { stage: "reflecting", state: card.status === "complete" ? "done" : "not_started", message: null },
    ],
    rubrics: [
      { rubric: "judge.verification", label: "Verification Discipline", conversation_id: "c1", rating: 0.75, review_status: "complete" },
    ],
    models: { proposal_writer: "GLM 5.3", proposal_evaluator: "DeepSeek V4.1 Flash" },
    config: {
      model_catalog_version: "models-1",
      rubric_catalog_version: "rubrics-1",
      judge_models: judges.map((judge) => judge.id),
      challenge_judge_models: [judges[0].id],
      proposal_model: "wandb:GLM-5.3",
      proposal_evaluator_model: judges[0].id,
      rubrics: [],
      candidate_budget: 3,
      force: false,
    },
    auto_run: true,
    error: null,
  };
}

const model = (id: string, label: string, provider: string, roles: string[]) => ({ id, label, provider, supported_roles: roles });

export function catalog(): Catalog {
  const all = ["judge", "proposal_writer", "proposal_evaluator"];
  return {
    format: 1,
    models: {
      catalog_version: "models-2",
      available_models: [
        model("wandb:DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash", "wandb", all),
        model("codex:gpt-5.6-sol", "GPT-5.6 Sol", "codex", ["judge"]),
        model("wandb:GLM-5.3", "GLM 5.3", "wandb", all),
        model("wandb:Kimi-K2.6", "Kimi K2.6", "wandb", all),
      ],
      recommended_proposal_model: "wandb:Kimi-K2.6",
      recommended_judges: ["wandb:Kimi-K2.6", "wandb:GLM-5.3"],
      recommended_challenge_judges: ["wandb:Kimi-K2.6"],
      proposal_evaluator_preferences: ["wandb:GLM-5.3"],
    },
    rubrics: { catalog_version: "rubrics-2", rubrics: [{ id: "judge.verification", label: "Verification Discipline" }] },
  };
}

export function sessions(): SessionList {
  return {
    sessions: [
      { conversation_id: "conv-review-stack", turn_count: 114, last_activity: "2026-11-26T16:00:00+00:00", input_preview: "Review this stack of PRs" },
      { conversation_id: "01a1ad64-7d00-75f7-b10f-e2e0471fdf58", turn_count: 351, last_activity: "2026-11-05T22:00:00+00:00", input_preview: null },
    ],
    total: 2,
    truncated: false,
  };
}

export function providerCheck(): ProviderCheck {
  return {
    format: 1,
    checked_at: "2026-11-26T18:00:00+00:00",
    providers: [
      { id: "claude", title: "Claude", state: "signed_in", detail: null },
      { id: "codex", title: "Codex", state: "signed_out", detail: "Not logged in" },
      { id: "devin", title: "Devin", state: "signed_in", detail: null },
      { id: "wandb", title: "W&B Inference", state: "signed_in", detail: null },
      { id: "openai", title: "OpenAI", state: "no_key", detail: "OPENAI_API_KEY is not set" },
    ],
  };
}
