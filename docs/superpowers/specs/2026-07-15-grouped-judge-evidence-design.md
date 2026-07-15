# Grouped Judge Evidence

## Goal

Preserve every useful judge observation while keeping trace citations unique,
exact, and auditable. A judge must not fail merely because several distinct
observations cite the same allowed trace.

## Product fit

Monopole treats trace identity as authoritative and model judgments as bounded
opinions over supplied evidence. Grouping observations by trace keeps that
boundary intact: it does not invent evidence, loosen the evidence allowlist,
coerce scores, or change reviewer aggregation. It prevents a representation
mismatch from discarding an otherwise valid judgment and unnecessarily
escalating selective review.

This change is confined to the judge-verdict boundary. GEPA remains downstream:
it consumes persisted evaluation feedback during reflection and does not merge
evidence observations inside individual verdicts.

## Verdict contract

Judge verdict schema version 3 represents evidence as unique trace groups:

```json
{
  "schema_version": 3,
  "status": "scored",
  "score": 0.5,
  "rationale": "The sequence used suitable tools but included avoidable work.",
  "evidence": [
    {
      "id": "trace-1",
      "observations": [
        "Installation began before repository requirements were inspected.",
        "Repository exploration was subsequently thorough.",
        "Several environment checks were redundant."
      ]
    }
  ]
}
```

The canonical verdict has these invariants:

- each evidence ID appears exactly once;
- each ID belongs to the request's exact evidence allowlist;
- each group contains at least one nonblank observation;
- distinct observations retain first-seen order;
- exact duplicate observations within a group are retained once;
- scored verdicts contain at least one evidence group; and
- insufficient-evidence verdicts retain a null score and empty evidence.

Unknown IDs, blank observations, invalid scores, extra fields, and malformed
verdicts continue to fail closed.

## Boundary normalization

Structured output requests the grouped version-3 shape. As defense against a
model emitting more than one group for the same allowed trace, the authoritative
verdict parser merges those groups before constructing the canonical verdict.
The merge is deterministic and lossless for distinct observations; it neither
adds evidence nor changes the score or rationale.

Normalization belongs in the shared verdict parser rather than a provider
adapter. Claude, Codex, OpenAI, and W&B outputs therefore receive identical
validation, and no provider-specific retry or compatibility layer is added.

## Audit data

Successful and abstained reviewer attempts retain bounded canonical evidence
groups alongside the existing unique `evidence_ids`, rationale, score, model,
usage, output mode, and content digest. Failed attempts continue to retain only
safe bounded error information. Raw model output and prompts remain excluded.

Persisted observation text follows the same trust treatment as persisted judge
rationales: it is model-authored structured outcome text, bounded before it
enters run progress or feedback metadata.

## Versioning and scope

The verdict schema version advances from 2 to 3, and the judge prompt contract
changes with it. The effective run configuration already pins prompt and
pipeline compatibility, so a run started under an incompatible contract must
fail closed rather than mix verdict versions. The product is pre-release and
current-contract-only; no legacy run-row migration or version-2 adapter is
introduced.

No changes are made to rubric applicability, evidence selection, reviewer
escalation, score mean-pooling, analysis aggregation, GEPA optimization,
reflection ranking, or promotion.

## Testing

Tests will establish the behavior in this order:

1. A version-3 verdict with two groups for one allowed ID initially fails under
   the existing parser, reproducing the Claude failure.
2. Parsing returns one canonical group containing both distinct observations in
   first-seen order.
3. Exact duplicate observations are retained once.
4. Unknown IDs and blank observations remain rejected.
5. Judge execution records one unique evidence ID and the complete bounded
   grouped observations without raw output.
6. Selective review treats the normalized verdict as a successful attempt and
   does not escalate solely because its source groups repeated an allowed ID.
7. Focused judge tests, the backend test suite, Ruff checks, and formatting
   checks pass.

## Cleanup criteria

The version-2 prompt layout and uniqueness-rejection test are replaced rather
than retained as parallel behavior. No provider-specific duplicate handling,
retry path, legacy schema adapter, or temporary diagnostic logging remains.
The lasting verdict invariant is consolidated into `specs/02-evaluation.md`;
this temporary design artifact can then be removed cleanly.
