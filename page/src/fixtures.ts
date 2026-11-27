import type { Card, Catalog, ModelDescriptor, ProviderCheck, RunDetail, SessionEstimates, SessionList } from "./protocol";

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

type Role = ModelDescriptor["supported_roles"][number];
const model = (id: string, label: string, provider: string, family: string, roles: Role[], maxInput = 200_000): ModelDescriptor => ({
  id,
  label,
  provider,
  provider_model: id.split(":")[1],
  family,
  supported_roles: roles,
  max_input_tokens: maxInput,
  token_counter: "utf8_bytes_div_3",
  window_budget_tokens: Math.round(maxInput * 0.6),
  merge_budget_tokens: 20_000,
  question_budget_tokens: 4_000,
  whole_budget_tokens: null,
});

export function catalog(): Catalog {
  const all: Role[] = ["judge", "proposal_writer", "proposal_evaluator"];
  return {
    format: 1,
    models: {
      catalog_version: "models-2",
      available_models: [
        model("wandb:DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash", "wandb", "deepseek", all),
        // A small context, which the large session doesn't fit.
        model("codex:gpt-5.6-sol", "GPT-5.6 Sol", "codex", "openai", ["judge"], 64_000),
        model("wandb:GLM-5.3", "GLM 5.3", "wandb", "glm", all),
        model("wandb:Kimi-K2.6", "Kimi K2.6", "wandb", "kimi", all),
      ],
      judging_context: {
        contract_version: "4",
        large_model_threshold_tokens: 200000,
        digest_max_tokens: 1_000,
        finding_max_tokens: 10_000,
        overlap_turns: 1,
        max_chunks: 40,
      },
      recommended_proposal_model: "wandb:Kimi-K2.6",
      recommended_judges: ["wandb:Kimi-K2.6", "wandb:GLM-5.3"],
      recommended_challenge_judges: ["wandb:Kimi-K2.6"],
      proposal_evaluator_preferences: ["wandb:GLM-5.3"],
    },
    rubrics: {
      catalog_version: "rubrics-2",
      rubrics: [
        { id: "judge.verification", label: "Verification Discipline", version: "1", content_digest: "sha256:0", evaluation_unit: "session", pass_threshold: 0.7 },
      ],
    },
  };
}

/** Judging token estimates for the two sessions: the review stack has one turn larger than GPT-5.6
 * Sol's whole context. */
export function sessionEstimates(): SessionEstimates {
  const estimate = (turns: number[]) => ({
    utf8_bytes_div_3: { turn_tokens: turns, largest_turn_tokens: Math.max(...turns), total_tokens: turns.reduce((a, b) => a + b, 0) },
  });
  return {
    "conv-review-stack": estimate([20_000, 90_000, 15_000]),
    "01a1ad64-7d00-75f7-b10f-e2e0471fdf58": estimate([4_000, 6_000]),
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
