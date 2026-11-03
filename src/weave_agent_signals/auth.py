"""The W&B API key, which authorizes CES and W&B Inference."""

from __future__ import annotations

import netrc
import os


def get_api_key() -> str:
    """`WANDB_API_KEY`, else the `api.wandb.ai` entry in `~/.netrc`."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
            if auth:
                return auth[2]
        except Exception:
            pass
    raise RuntimeError("No W&B API key found. Set WANDB_API_KEY or add api.wandb.ai to ~/.netrc")
