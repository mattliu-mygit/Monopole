from dataclasses import dataclass

CATALOG_VERSION = "v1"
ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)
RECOMMENDATION_THRESHOLD = 0.5
SAMPLING_RATE = 1.0
DEFAULT_MODEL = "openai/gpt-4.1-mini"
TURN_OP_NAME = "weave.genai.turn_ended"

_BASE_PROMPT = """You are a high recall production monitor. Judge only the visible current agent turn.
Return one JSON object with keys rating and reason. rating must be exactly one of 0.0, 0.25, 0.5, 0.75, or 1.0, where 1.0 means no evidence of the problem, 0.75 means weak or ambiguous indication, 0.5 means plausible indication, 0.25 means strong indication, and 0.0 means explicit or severe evidence. reason must be at most 240 characters and cite visible turn evidence without quoting secrets. Do not infer unseen conversation history.

User messages:
{input_messages}

Agent messages:
{output_messages}

Rubric:
{rubric}
"""


@dataclass(frozen=True)
class SignalDefinition:
    slug: str
    version: str
    description: str
    rubric: str

    @property
    def monitor_name(self) -> str:
        return f"agent-signal-{self.slug}-{self.version}"

    @property
    def scorer_name(self) -> str:
        return f"agent-signal-{self.slug}-{self.version}-judge"

    @property
    def scoring_prompt(self) -> str:
        return _BASE_PROMPT.format(
            input_messages="{input_messages}",
            output_messages="{output_messages}",
            rubric=self.rubric,
        )


CATALOG = (
    SignalDefinition(
        "user-frustration",
        CATALOG_VERSION,
        "User expresses annoyance, impatience, confusion, or dissatisfaction.",
        "Rate evidence that the user expresses annoyance, impatience, confusion, or dissatisfaction in this turn. Do not treat a neutral correction or ordinary follow-up as frustration without affective evidence.",
    ),
    SignalDefinition(
        "user-correction-or-rejection",
        CATALOG_VERSION,
        "User explicitly corrects, rejects, or redirects prior agent work.",
        "Rate evidence that the user says prior agent work is wrong, rejects it, or redirects the agent in order to correct it. A new preference without rejection is not enough.",
    ),
    SignalDefinition(
        "explicit-repeat-or-rephrase-cue",
        CATALOG_VERSION,
        "User explicitly indicates that an unresolved request is being repeated or restated.",
        "Rate explicit language showing the user is repeating, restating, or asking again for an unresolved request. Do not infer semantic repetition when the current turn has no repeat cue.",
    ),
    SignalDefinition(
        "stalled-or-deferred-response",
        CATALOG_VERSION,
        "Agent unnecessarily hands work back, postpones action, or stops early.",
        "Rate evidence that the agent unnecessarily asks the user to do available work, postpones action, or stops despite having enough visible information and authority to continue. Necessary approval or missing critical input is not a stall.",
    ),
    SignalDefinition(
        "low-quality-response",
        CATALOG_VERSION,
        "Agent response is materially irrelevant, evasive, repetitive, or incomplete.",
        "Rate evidence that the agent response fails the visible request because it is materially irrelevant, evasive, repetitive, plainly incomplete, or unsupported. Minor style issues alone are not enough.",
    ),
)
