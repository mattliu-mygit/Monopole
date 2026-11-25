from __future__ import annotations

import pytest

from weave_agent_signals.run_config import JudgingContextPolicy


@pytest.mark.parametrize(
    ("limit", "prompt", "generation", "safety"),
    [
        (8_192, 2_000, 1_000, 1_000),
        (32_800, 4_920, 3_280, 1_640),
        (131_072, 6_000, 4_000, 6_554),
        (1_000_000, 6_000, 32_000, 8_000),
    ],
)
def test_reserve_parts_scale_with_model_size(limit, prompt, generation, safety):
    policy = JudgingContextPolicy()

    assert (
        policy.prompt_reserve(limit),
        policy.generation_budget(limit),
        policy.safety_reserve(limit),
    ) == (prompt, generation, safety)
    assert policy.capacity_reserve(limit) == prompt + generation + safety
