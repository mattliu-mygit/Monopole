"""Shared validation for timezone-aware HTTP date filters."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DateFilterError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def parse_selection_bounds(
    since: str | None,
    until: str | None,
    timezone_name: str | None,
) -> tuple[datetime | None, datetime | None]:
    if timezone_name is not None:
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise DateFilterError(
                "invalid_timezone",
                f"Unknown IANA timezone: {timezone_name}",
            ) from error

    def parse(value: str | None) -> datetime | None:
        if value is None or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise DateFilterError(
                "invalid_datetime",
                f"Cannot parse datetime: {value}",
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise DateFilterError(
                "invalid_datetime",
                "Selection boundaries must include a timezone offset",
            )
        return parsed

    since_value = parse(since)
    until_value = parse(until)
    if since_value is not None and until_value is not None and since_value > until_value:
        raise DateFilterError("invalid_date_range", "since must not be after until")
    return since_value, until_value
