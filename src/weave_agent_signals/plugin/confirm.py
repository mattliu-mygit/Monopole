"""Prompting Time's confirmation convention for the plugin's `confirm: true` actions.

`--print` saves the request in `<state folder>/<action>-request.json`, names that file in its
output, and prints `fingerprint: <sha256>` of it on its last line; nothing else happens.
`--expect <sha256>` runs only when the request, built again from what is current, still has that
fingerprint, so a request that changed after the user confirmed does not run; the action then exits
13, and Prompting Time previews the new request and asks again. The saved file only shows the
request: a later preview may replace it, so `--expect` never reads it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from weave_agent_signals.plugin.settings import state_dir, write_atomic


class Refused(Exception):
    """The action did nothing, for the reason in its message."""


class RequestChanged(Refused):
    """The request built now has another fingerprint than the one the user confirmed."""


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def request_text(action: str, requests: list[dict[str, Any]]) -> str:
    """The text the user confirms: one item's request, or for several items one request listing
    each, so a single fingerprint covers them all and a change to any of them changes it."""

    request = requests[0] if len(requests) == 1 else {"action": action, "runs": requests}
    return json.dumps(request, indent=2, ensure_ascii=False) + "\n"


def confirmed(action: str, text: str, *, print_only: bool, expect: str | None, what: str) -> bool:
    """With `print_only`, save `text` for the user to confirm and return False. Otherwise return
    True when `text` still has the fingerprint the user confirmed, and refuse when it does not.
    `what` names the request in messages, such as "delete run-1"."""

    request_file = state_dir() / f"{action}-request.json"
    if print_only:
        write_atomic(request_file, text)
        print(f"Saved the request to {what} in {request_file}; nothing ran yet.")
        print(f"fingerprint: {fingerprint(text)}")
        return False
    if expect != fingerprint(text):
        raise RequestChanged(f"The request to {what} changed after you confirmed; nothing ran.")
    # The file only showed the request; the run rebuilt it, so the file is no longer needed.
    request_file.unlink(missing_ok=True)
    return True


def add_confirm_flags(parser) -> None:
    """`--print` or `--expect <fingerprint>`, one of which Prompting Time always passes."""

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true", dest="print_only")
    mode.add_argument("--expect", metavar="FINGERPRINT")
