import type { JudgingAttemptSummary, JudgingPlan, JudgingProgress, ReflectingProgress, ReflectionResult, ReviewAttempt } from "../../frontend/src/types";
import type { Analysis, Card, Catalog, ModelDescriptor, ProviderCheck, RunDetail, SessionEstimates, SessionList } from "./protocol";

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

const SESSION = "01a1ad64-7d00-75f7-b10f-e2e0471fdf58";
const RUBRICS = [
  { id: "judge.verification", label: "Verification Discipline", version: "1", content_digest: "sha256:0", evaluation_unit: "session" as const, pass_threshold: 0.7 },
  { id: "judge.session_outcome", label: "Session Outcome Quality", version: "1", content_digest: "sha256:1", evaluation_unit: "session" as const, pass_threshold: 0.7 },
];
const CONTEXT = { contract_version: "4", digest_max_tokens: 1_000, finding_max_tokens: 10_000, large_model_threshold_tokens: 200000, max_chunks: 40, overlap_turns: 1 } as const;

function attempt(position: number, requested_model: string, status: ReviewAttempt["status"], extra: Partial<ReviewAttempt> = {}): ReviewAttempt {
  return {
    position,
    role: "judge",
    trigger: "panel",
    requested_model,
    requested_family: "deepseek",
    requested_backend: "wandb",
    resolved_model: requested_model,
    resolved_family: "deepseek",
    status,
    score: null,
    rationale: null,
    behavioral_feedback: null,
    error_type: null,
    message: null,
    evidence_ids: [],
    output_mode: "json_schema",
    schema_name: "session_verdict",
    verdict_schema_version: 2,
    schema_fallback_reason: null,
    raw_output_digest: null,
    transport_request_count: 1,
    usage: { input_tokens: 41_000, output_tokens: 900 },
    steps: [],
    ...extra,
  };
}

/** Judging that rated Verification Discipline 0.75 and failed Session Outcome Quality when
 * DeepSeek V4.1 Flash cited evidence IDs that don't exist. */
function judging(failed: boolean): JudgingProgress {
  const rated = attempt(1, "wandb:DeepSeek-V4.1-Flash", "succeeded", {
    score: 0.75,
    rationale: "Ran the tests after each change.",
    behavioral_feedback: { success: "Verified every fix with the test suite.", problem: null, desired_behavior: null },
    steps: [
      {
        phase: "window",
        reused: false,
        schema_name: "window_findings",
        artifact_id: "artifact-1",
        requested_model: "wandb:DeepSeek-V4.1-Flash",
        resolved_model: "wandb:DeepSeek-V4.1-Flash",
        transport_request_count: 1,
        usage: { input_tokens: 30_000 },
      },
    ],
  });
  const broken = attempt(1, "wandb:DeepSeek-V4.1-Flash", "failed", {
    error_type: "ValueError",
    message: 'unknown evidence IDs: ["e224"]',
    transport_request_count: 28,
  });
  const summaries: JudgingAttemptSummary[] = [
    { rubric: "judge.verification", scope: "session" as const, conversation_id: SESSION, rating: 0.75, review_status: "complete" as const, attempt_count: 1, successful_reviewer_count: 1, attempts: [rated] },
    ...(failed
      ? [{ rubric: "judge.session_outcome", scope: "session" as const, conversation_id: SESSION, rating: null, review_status: "failed" as const, attempt_count: 1, successful_reviewer_count: 0, attempts: [broken] }]
      : []),
  ];
  return {
    plan_id: "plan-1",
    phase: failed ? "judging_failed" : "judging_complete",
    status_message: failed ? "Session Outcome Quality failed" : "Judging complete",
    started_at: "2026-11-05T22:02:00+00:00",
    planned_rubrics: 2,
    rubrics_completed: failed ? 1 : 2,
    rated_rubrics: 1,
    not_evaluable_rubrics: 0,
    minimum_reviewer_attempts: 2,
    maximum_reviewer_attempts: 4,
    reviewer_attempts_completed: 2,
    digest_steps_completed: 3,
    maximum_digest_steps: 3,
    window_steps_completed: 5,
    maximum_window_steps: 6,
    merge_steps_completed: 1,
    maximum_merge_steps: 2,
    scores_written: 0,
    failure_count: failed ? 1 : 0,
    write_failure_count: 0,
    coverage_complete: !failed,
    events: [
      { id: 1, at: "2026-11-05T22:02:00+00:00", phase: "judging_started", message: "Judging started" },
      { id: 2, at: "2026-11-05T22:03:10+00:00", phase: "transport_retry", message: "Retrying DeepSeek V4.1 Flash", model: "wandb:DeepSeek-V4.1-Flash", request_attempt: 2, max_attempts: 3, provider_status: 429 },
      { id: 3, at: "2026-11-05T22:14:06+00:00", phase: "rubric_failed", message: "Session Outcome Quality failed", rubric: "judge.session_outcome" },
    ],
    attempt_summary_count: summaries.length,
    attempt_summaries_truncated: false,
    attempt_summaries: summaries,
    failure_detail_count: failed ? 1 : 0,
    failure_details_truncated: false,
    failure_details: failed
      ? [{ rubric: "judge.session_outcome", scope: "session", conversation_id: SESSION, error_type: "ReviewFailed", message: "1 judge failed", attempt_count: 1, attempts: [broken] }]
      : [],
  };
}

function judge(position: number, id: string, label: string, provider: string, family: string) {
  return { ...model(id, label, provider, family, ["judge"]), position, role: "judge" as const };
}

const planJudge = () => judge(1, "wandb:DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash", "wandb", "deepseek");

/** A plan of two windows over a five-turn session: turns 1–3, and 4–5 with turn 3 for context. */
function plan(): JudgingPlan {
  const turns = [1, 2, 3, 4, 5].map((position) => ({ trace_id: `t${position}`, position, estimated_tokens: 9_000, raw_digest: `d${position}` }));
  return {
    plan_id: "plan-1",
    cohort_id: "sha256:cohort",
    schema_version: "1",
    panel_size: 1,
    protocol: {},
    input_policy: CONTEXT,
    requested_rubrics: RUBRICS,
    totals: { sessions_planned: 1, turns_considered: 5, windows_planned: 2, planned_rubrics: 2, minimum_reviewer_attempts: 2, maximum_reviewer_attempts: 4, maximum_digest_calls: 3, maximum_window_calls: 6, maximum_merge_calls: 2 },
    sessions: [
      {
        conversation_id: SESSION,
        turn_count: 5,
        raw_coverage_trace_ids: turns.map((turn) => turn.trace_id),
        rubrics: [],
        reviewers: [
          {
            ordinal: 1,
            judge: planJudge(),
            status: "planned",
            skip_reason: null,
            work_bounds: { digest_calls: 3, window_calls_per_rubric: 2, merge_calls_per_rubric: 1 },
            window_plan: {
              plan_id: "window-plan-1",
              contract_version: "3",
              conversation_id: SESSION,
              token_counter: "utf8_bytes_div_3",
              input_cap_tokens: 200_000,
              capacity_reserve_tokens: 4_000,
              raw_budget_tokens: 120_000,
              target_raw_tokens: 27_000,
              merge_input_tokens: 20_000,
              overlap_turns: 1,
              chunk_count: 2,
              raw_coverage_trace_ids: turns.map((turn) => turn.trace_id),
              raw_turns: turns,
              windows: [
                { window_id: "w1", index: 1, core_trace_ids: ["t1", "t2", "t3"], raw_trace_ids: ["t1", "t2", "t3"], raw_tokens: 27_000, raw_turn_digests: [] },
                { window_id: "w2", index: 2, core_trace_ids: ["t4", "t5"], raw_trace_ids: ["t3", "t4", "t5"], raw_tokens: 27_000, raw_turn_digests: [] },
              ],
            },
          },
        ],
      },
    ],
  };
}

function bundle(revision: string, content: string) {
  return {
    revision,
    scope: null,
    targets: [{ locator: "outcome/CLAUDE.md", display_name: "CLAUDE.md", kind: "file", path: "CLAUDE.md", exists: true, content, revision }],
  };
}

const evaluation = (target: string, score: number, rationale: string) => ({
  evaluation_id: `eval-${target}`,
  target_revision: target,
  score,
  rationale,
  requested_model: "wandb:DeepSeek-V4.1-Flash",
  requested_family: "deepseek",
  requested_backend: "wandb",
  resolved_model: "wandb:DeepSeek-V4.1-Flash",
  resolved_family: "deepseek",
  resolved_backend: "wandb",
  usage: { input_tokens: 12_000 },
});

/** Reflection whose best proposal was predicted 0.35 against the current 0.15, but whose A/B
 * check stopped when the sandbox's preflight failed. */
function reflection(): { reflecting_progress: ReflectingProgress; reflecting_result: ReflectionResult } {
  const writer = model("wandb:GLM-5.3", "GLM 5.3", "wandb", "glm", ["proposal_writer"]);
  const candidate = (id: string, score: number, rationale: string) => ({
    candidate_id: id,
    generation_attempt_id: `attempt-${id}`,
    bundle: bundle(`rev-${id}`, "Run the tests."),
    evaluation: evaluation(`rev-${id}`, score, rationale),
    rationale,
    requested_writer: writer,
    resolved_writer_model: "wandb:GLM-5.3",
    resolved_writer_family: "glm",
    resolved_writer_backend: "wandb",
    score,
    score_delta: Number((score - 0.15).toFixed(4)),
  });
  return {
    reflecting_progress: {
      phase: "complete",
      status_message: "Reflection complete",
      started_at: "2026-11-26T16:16:00+00:00",
      proposal_writer: "GLM 5.3",
      proposal_evaluator: "DeepSeek V4.1 Flash",
      no_improvement_patience: 2,
      total_attempts: 3,
      attempted: 3,
      valid: 2,
      rejected: 1,
      scored: 2,
      events: [
        { id: 1, at: "2026-11-26T16:16:00+00:00", phase: "reflection_started", message: "Reflection started" },
        { id: 2, at: "2026-11-26T16:18:00+00:00", phase: "candidate_rejected", message: "Proposal attempt 2 was rejected", candidate: 2, changed_paths: ["outcome/AGENTS.md"], response_excerpt: "I edited a file outside the scope." },
        { id: 3, at: "2026-11-26T16:21:00+00:00", phase: "candidate_scored", message: "Proposal 1 scored", candidate: 1, score: 0.35 },
      ],
    },
    reflecting_result: {
      score_basis: "predicted_evaluator",
      baseline: bundle("rev-base", "Be careful."),
      baseline_score: 0.15,
      baseline_evaluation: evaluation("rev-base", 0.15, "The current bundle is long."),
      baseline_won: true,
      reason: null,
      recommended_candidate_id: null,
      provisional_candidate_id: "candidate-a",
      candidates: [candidate("candidate-a", 0.35, "Targets outcome.command and outcome.test."), candidate("candidate-b", 0.2, "Small gain.")],
      generation_attempts: [
        {
          attempt_id: "attempt-2",
          number: 2,
          status: "failed",
          candidate_revision: null,
          changed_paths: ["outcome/AGENTS.md"],
          error_type: "ScopeViolation",
          error: "changed a file outside the managed scope",
          requested_writer: writer,
          resolved_model: "wandb:GLM-5.3",
          resolved_family: "glm",
          resolved_backend: "wandb",
          response_digest: null,
          response_excerpt: "I edited a file outside the scope.",
          usage: {},
        },
      ],
      challenge: {
        challenge_id: "challenge-1",
        schema_version: "1",
        candidate_id: "candidate-a",
        status: "incomplete",
        winner: "tie",
        reason: "Sandbox preflight failed",
        task: null,
        baseline: null,
        candidate: null,
        judges: [],
        execution: {
          execution_id: "exec-1",
          schema_version: "1",
          command: ["codex", "exec"],
          effort: null,
          environment: [],
          harness: "codex",
          harness_version: "0.98",
          image: "monopole-sandbox",
          image_digest: "sha256:image",
          model: "gpt-5.6-sol",
          model_family: "openai",
          network_enabled: false,
          runtime_files: [],
          timeout_seconds: 900,
          workspace_digest: "sha256:workspace",
        },
      },
    },
  };
}

/** The run `run --item` prints for a card: the card, its stage summary, and the web UI's run. */
export function detail(card: Card): RunDetail {
  const failed = card.status === "failed";
  const reflected = card.status === "complete";
  const configured = card.status !== "created";
  return {
    ...card,
    shortened: false,
    current_stage_succeeded: !failed && card.group !== "running",
    auto_run: true,
    error: failed ? "1 rubric(s) failed and 0 were not evaluable" : null,
    data_selection: { session_ids: [SESSION], since: null, until: null, timezone: null },
    run_config: {
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
    effective_config: configured
      ? {
          model_catalog_version: "models-1",
          rubric_catalog_version: "rubrics-1",
          pipeline_version: "7",
          candidate_budget: 3,
          force: false,
          judging_context: CONTEXT,
          schema_version: "5",
          rubrics: RUBRICS,
          selection_warnings: [],
          models: {
            judges: [planJudge(), judge(2, "codex:gpt-5.6-sol", "GPT-5.6 Sol", "codex", "openai")],
            challenge_judges: [planJudge()],
            proposal_writer: model("wandb:GLM-5.3", "GLM 5.3", "wandb", "glm", ["proposal_writer"]),
            proposal_evaluator: model("wandb:DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash", "wandb", "deepseek", ["proposal_evaluator"]),
          },
        }
      : null,
    turn_cohort: configured ? { cohort_id: "sha256:cohort", pinned_at: "2026-11-05T22:01:00+00:00", turn_count: card.turns, session_count: 1 } : null,
    judging_plan: card.status === "scoring" ? null : plan(),
    scoring_progress: null,
    scoring_result: {
      turns_scored: 2,
      sessions_scored: 1,
      scores_written: 12,
      errors: 0,
      status_message: "Scoring complete: 12 scores written.",
      turn_details: [
        { trace_id: "t1", conversation_id: SESSION, user_input: card.title, model: "claude-sonnet-5", tokens: 18_000, tool_count: 14, tools_used: ["Bash"], errors: 0, denials: 0, steering: 0, scores: { "outcome.tests_passed": 1, efficiency: 0.62 } },
      ],
    },
    judging_progress: card.group === "running" ? { ...judging(false), status_message: "Judging 2 of 6 rubrics", attempt_summaries: [], rubrics_completed: 0 } : null,
    judging_result: card.group === "running" || card.status === "scoring" ? null : judging(failed),
    ...(reflected ? reflection() : { reflecting_progress: null, reflecting_result: null }),
    reflection_review: reflected ? { status: "pending", selected_candidate_id: "candidate-a", draft: null } : null,
    reflection_review_revision: reflected ? 1 : 0,
    stages: [
      { stage: "scoring", state: "done", message: "Scoring complete: 12 scores written." },
      { stage: "judging", state: failed ? "failed" : card.group === "running" ? "running" : card.status === "scoring" ? "not_started" : "done", message: card.failure?.cause ?? null },
      { stage: "reflecting", state: reflected ? "done" : "not_started", message: null },
    ],
    rubrics: [{ rubric: "judge.verification", label: "Verification Discipline", conversation_id: SESSION, rating: 0.75, review_status: "complete" }],
    models: { proposal_writer: "GLM 5.3", proposal_evaluator: "DeepSeek V4.1 Flash" },
  };
}

/** The web UI's analysis: tests passing fell from 0.90 to 0.70 over the last week. */
export function analysis(): Analysis {
  return {
    summary: [
      { scorer: "efficiency", count: 120, mean: 0.62, ci: [0.58, 0.66], binary: false, pass_rate: null, tag_counts: {}, confident: true },
      { scorer: "outcome.tests_passed", count: 40, mean: 0.8, ci: [0.7, 0.9], binary: true, pass_rate: 0.8, tag_counts: {}, confident: true },
      { scorer: "judge.verification", count: 4, mean: 0.75, ci: [0.4, 1], binary: false, pass_rate: null, tag_counts: {}, confident: false },
    ],
    ab_leaderboard: [
      { config_version: "a1b2c3d4e5", evaluated_target_count: 30, scores: { efficiency: { mean: 0.6, count: 60 }, "outcome.tests_passed": { mean: 0.9, count: 20 } } },
      { config_version: "f6e5d4c3b2", evaluated_target_count: 28, scores: { efficiency: { mean: 0.64, count: 60 }, "outcome.tests_passed": { mean: 0.7, count: 20 } } },
    ],
    trends: [{ scorer: "outcome.tests_passed", direction: "regression", older_mean: 0.9, recent_mean: 0.7, delta: -0.2, significant: true, sample_count: 40 }],
    coaching_markdown: "Tests passed less often this week. Run the suite before ending a turn.",
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
      // GPT-5.6 Sol is recommended, as the real catalog recommends Codex, but the large session
      // doesn't fit it.
      recommended_judges: ["wandb:Kimi-K2.6", "codex:gpt-5.6-sol"],
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
