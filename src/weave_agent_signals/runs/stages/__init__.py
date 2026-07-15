"""Shared evaluation-stage contracts."""


class StageCancelled(Exception):
    """Signal cooperative cancellation without treating it as a stage failure."""
