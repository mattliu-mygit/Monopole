"""Raw evidence container shared by window planning and rendering."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeDigest:
    text: str
    evidence_ids: tuple[str, ...]
