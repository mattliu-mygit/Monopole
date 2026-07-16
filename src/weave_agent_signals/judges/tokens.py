"""Token counting for model-specific judging capacity."""

from typing import Literal

import tiktoken

TokenCounterName = Literal["utf8_bytes_div_3", "o200k_base", "o200k_harmony"]


def count_tokens(text: str, counter: TokenCounterName) -> int:
    if not text:
        return 0
    if counter == "utf8_bytes_div_3":
        return (len(text.encode("utf-8")) + 2) // 3
    return len(tiktoken.get_encoding(counter).encode(text))
