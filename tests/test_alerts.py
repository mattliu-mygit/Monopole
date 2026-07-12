"""Tests for the alerting sinks and dedup used by the monitor command."""

from __future__ import annotations

import httpx

from weave_agent_signals import alerts


def test_trend_alert_key_and_text():
    reg = {
        "scorer": "outcome.test",
        "direction": "regression",
        "older_mean": 0.8,
        "recent_mean": 0.3,
        "delta": -0.5,
        "sample_count": 20,
    }
    a = alerts.trend_alert(reg)
    assert a.key == "trend:outcome.test:regression"
    assert "outcome.test" in a.text and "0.80" in a.text and "0.30" in a.text


def test_config_alert_key_and_text():
    reg = {
        "scorer": "outcome.git",
        "config": "abc123",
        "prev_config": "def456",
        "new_mean": 0.2,
        "prev_mean": 0.9,
        "delta": -0.7,
        "sample_count": 8,
    }
    a = alerts.config_alert(reg)
    assert a.key == "config:abc123:outcome.git"
    assert "abc123" in a.text and "def456" in a.text


def test_send_logs_and_posts_webhook(monkeypatch):
    posted = {}

    def fake_post(url, json, timeout):
        posted["url"] = url
        posted["text"] = json["text"]
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", fake_post)
    delivered = alerts.send(
        [alerts.Alert("k", "something dropped")], webhook="https://hook.example/x"
    )
    assert posted["url"] == "https://hook.example/x"
    assert "something dropped" in posted["text"]
    assert len(delivered) == 1  # a successful POST counts as delivered


def test_send_failed_webhook_not_delivered(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(alerts.httpx, "post", boom)
    delivered = alerts.send([alerts.Alert("k", "x")], webhook="https://hook.example/x")
    assert delivered == []  # not delivered → caller won't dedup it


def test_send_rejected_webhook_not_delivered(monkeypatch):
    def reject(url, json, timeout):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", reject)
    delivered = alerts.send([alerts.Alert("k", "x")], webhook="https://hook.example/x")
    assert delivered == []  # 5xx is a failure even though httpx doesn't raise


def test_send_no_webhook_delivers_via_log():
    delivered = alerts.send([alerts.Alert("k", "x"), alerts.Alert("k2", "y")])
    assert len(delivered) == 2  # log-only always delivers


def test_seen_roundtrip(tmp_path):
    p = str(tmp_path / "seen.json")
    assert alerts.load_seen(p) == set()  # missing file → empty
    alerts.save_seen(p, {"a", "b"})
    assert alerts.load_seen(p) == {"a", "b"}


def test_load_seen_none_path():
    assert alerts.load_seen(None) == set()


def test_load_seen_non_list_json_is_empty(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text('{"not": "a list"}')  # hand-corrupted into an object
    assert alerts.load_seen(str(p)) == set()


def test_save_seen_creates_missing_parent_dir(tmp_path):
    p = str(tmp_path / "nested" / "dir" / "seen.json")
    alerts.save_seen(p, {"a"})  # parent dirs don't exist yet
    assert alerts.load_seen(p) == {"a"}


def test_save_seen_bad_path_does_not_raise(tmp_path):
    # a path whose parent is a file, not a dir → OSError, must be swallowed
    afile = tmp_path / "afile"
    afile.write_text("x")
    alerts.save_seen(str(afile / "seen.json"), {"a"})  # must not raise
