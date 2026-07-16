import pytest
import tiktoken

from weave_agent_signals.judges.tokens import count_tokens


def test_count_tokens_returns_zero_for_empty_text():
    assert count_tokens("", "utf8_bytes_div_3") == 0
    assert count_tokens("", "o200k_base") == 0
    assert count_tokens("", "o200k_harmony") == 0


def test_count_tokens_uses_rounded_up_utf8_byte_fallback():
    assert count_tokens("abcd", "utf8_bytes_div_3") == 2
    assert count_tokens("é", "utf8_bytes_div_3") == 1
    assert count_tokens("🙂", "utf8_bytes_div_3") == 2


def test_count_tokens_uses_named_tiktoken_encoding():
    text = "Capacity-aware judging 🙂"

    for counter in ("o200k_base", "o200k_harmony"):
        expected = len(tiktoken.get_encoding(counter).encode(text))
        assert count_tokens(text, counter) == expected


@pytest.mark.parametrize(
    ("counter", "text"),
    [
        ("o200k_base", "captured <|endoftext|> evidence"),
        ("o200k_harmony", "captured <|endoftext|> evidence"),
        ("o200k_harmony", "captured <|startoftext|> evidence"),
    ],
)
def test_count_tokens_treats_special_token_text_as_captured_evidence(counter, text):
    first = count_tokens(text, counter)

    assert first > 0
    assert count_tokens(text, counter) == first
