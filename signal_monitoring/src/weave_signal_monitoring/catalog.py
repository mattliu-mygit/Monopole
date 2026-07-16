from dataclasses import dataclass

CATALOG_VERSION = "v1"
ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)
RECOMMENDATION_THRESHOLD = 0.5
SAMPLING_RATE = 1.0
DEFAULT_MODEL = "coreweave/openai/gpt-oss-20b"
TURN_OP_NAME = "weave.genai.turn_ended"

_BASE_PROMPT = """You are a high recall evaluation rater for an AI agent. Judge only the visible current agent turn.

Rating criterion:
{rubric}

Trace:
<agent>
  <name>{agent_name}</name>
  <version>{agent_version}</version>
  <description>{agent_description}</description>
</agent>
<conversation_name>{conversation_name}</conversation_name>
<system_instructions>
{system_instructions}
</system_instructions>
<input_messages>
{input_messages}
</input_messages>
<output_messages>
{output_messages}
</output_messages>
<status>
  <code>{status_code}</code>
  <message>{status_message}</message>
</status>

Respond with a JSON object of the form:

{"value": <float between 0.0 and 1.0>, "confidence": 0.0-1.0, "reason": "one short sentence citing the evidence"}

Rules:
- "value" MUST be a JSON number and exactly one of 0.0, 0.25, 0.5, 0.75, or 1.0.
- 1.0 means no evidence of the problem; 0.75 weak or ambiguous indication; 0.5 plausible indication; 0.25 strong indication; and 0.0 explicit or severe evidence.
- "confidence" is your certainty in the rating from 0.0 to 1.0.
- "reason" must be at most 240 characters and cite specific visible turn evidence without quoting secrets.
- Do not infer unseen conversation history.
- Do NOT emit markdown fences or commentary, only the JSON object.
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
        return f"{self.monitor_name}-scorer"

    @property
    def model_name(self) -> str:
        return f"{self.monitor_name}-model"

    @property
    def scoring_prompt(self) -> str:
        return _BASE_PROMPT.replace("{rubric}", self.rubric)


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
        "Evaluate only the user messages. Rate explicit language showing the user is repeating, restating, or asking again for an unresolved request. Phrases such as 'I already asked', 'again', or 'as I said' are direct repeat cues. Do not infer semantic repetition when the user messages contain no repeat cue.",
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
