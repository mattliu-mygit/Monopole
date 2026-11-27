"""The Tools actions run the CLI's own commands; here those commands are fakes, so no test
reads CES, writes Scores, or posts a webhook."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals import ces_commands, cli
from weave_agent_signals.cli import main
from weave_agent_signals.plugin import tools
from weave_agent_signals.plugin.settings import state_dir


@pytest.fixture
def ces(monkeypatch, state, tmp_path):
    monkeypatch.setenv("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", str(tmp_path / "none.json"))
    monkeypatch.setenv("WEAVE_AGENT_SIGNALS_CES_SCOPE_REF", "scope-1")
    monkeypatch.setenv("WEAVE_AGENT_SIGNALS_CES_DATASET_ID", "dataset-1")
    calls: list = []
    return calls


def _fingerprint(capsys, argv: list[str]) -> str:
    assert main(argv + ["--print"]) == 0
    out = capsys.readouterr().out
    found = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert found
    return found.group(1)


def test_score_confirms_then_runs_the_score_command_and_records_it(ces, monkeypatch, capsys):
    def fake_score(args):
        ces.append(args)
        print("Scored 3 turns, 1 sessions, 0 subagents (12 scores). Wrote 12.")
        return 0

    monkeypatch.setattr(ces_commands, "score", fake_score)
    fingerprint = _fingerprint(capsys, ["plugin", "score", "--hours", "168"])
    request = json.loads((state_dir() / "score-request.json").read_text())
    assert (request["hours"], request["scope"], request["dataset"]) == (168, "scope-1", "dataset-1")
    assert ces == []

    assert main(["plugin", "score", "--hours", "168", "--expect", fingerprint]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "done" and printed["exit_code"] == 0
    assert printed["output"] == "Scored 3 turns, 1 sessions, 0 subagents (12 scores). Wrote 12."
    since = datetime.now(timezone.utc) - timedelta(hours=168)
    assert abs((ces[0].since - since).total_seconds()) < 60
    assert (ces[0].limit, ces[0].force) == (100, False)
    assert tools.last_runs()["tools"]["score"]["state"] == "done"


def test_backfill_runs_whole_days_and_refuses_an_end_before_its_start(ces, monkeypatch, capsys):
    monkeypatch.setattr(ces_commands, "backfill", lambda args: ces.append(args) or 0)
    argv = ["plugin", "backfill", "--start", "2026-10-01", "--end", "2026-10-31"]
    fingerprint = _fingerprint(capsys, argv)

    assert main(argv + ["--expect", fingerprint]) == 0

    assert ces[0].start == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert ces[0].end == datetime(2026, 11, 1, tzinfo=timezone.utc)
    capsys.readouterr()
    argv = ["plugin", "backfill", "--start", "2026-10-31", "--end", "2026-10-01", "--print"]
    assert main(argv) == 1
    assert "--end must be on or after --start" in capsys.readouterr().err


def test_monitor_posts_to_the_settings_webhook_and_keeps_its_state_in_the_state_folder(
    ces, monkeypatch, capsys
):
    def fake_monitor(args):
        ces.append(args)
        logging.getLogger("weave_agent_signals.monitor").warning("ALERT trend down: outcome.test")
        print("1 new significant regression(s) alerted.")
        return 0

    monkeypatch.setattr(cli, "cmd_monitor", fake_monitor)
    main(["plugin", "save-settings", json.dumps({"alert_webhook": "https://hooks.example/abc"})])
    capsys.readouterr()
    fingerprint = _fingerprint(capsys, ["plugin", "monitor"])
    request = json.loads((state_dir() / "monitor-request.json").read_text())
    assert request["webhook_host"] == "hooks.example"

    assert main(["plugin", "monitor", "--expect", fingerprint]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] == (
        "ALERT trend down: outcome.test\n1 new significant regression(s) alerted."
    )
    assert ces[0].alert_webhook == "https://hooks.example/abc"
    assert ces[0].state_file == str(state_dir() / "monitor-state.json")


def test_insights_only_read_and_need_no_confirmation(ces, monkeypatch, capsys):
    def fake_analyze(args):
        ces.append(args)
        print("Judge agreement ...")
        return 0

    monkeypatch.setattr(cli, "cmd_analyze", fake_analyze)

    assert main(["plugin", "insights", "--kind", "judges"]) == 0

    assert json.loads(capsys.readouterr().out)["output"] == "Judge agreement ..."
    args = ces[0]
    assert (args.judges, args.subagents, args.segments, args.ab, args.coaching) == (
        True,
        False,
        None,
        False,
        False,
    )


def test_a_failing_tool_is_recorded_as_failed_with_its_error(ces, monkeypatch, capsys):
    def broken(args):
        raise RuntimeError("CES answered 503")

    monkeypatch.setattr(cli, "cmd_analyze", broken)

    assert main(["plugin", "insights", "--kind", "segments"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert (printed["state"], printed["exit_code"]) == ("failed", 1)
    assert printed["output"] == "RuntimeError: CES answered 503"


def test_a_running_tool_is_not_started_twice_and_a_dead_one_reads_as_stopped(ces, capsys):
    import os

    state_dir().mkdir(parents=True, exist_ok=True)
    running = {"state": "running", "pid": os.getpid(), "started_at": "x"}
    gone = {"state": "running", "pid": 999_999_999, "started_at": "x"}
    (state_dir() / "tools.json").write_text(json.dumps({"insights": running, "score": gone}))

    assert main(["plugin", "insights", "--kind", "judges"]) == 1
    assert "insights is already running" in capsys.readouterr().err
    assert main(["plugin", "tools"]) == 0
    assert json.loads(capsys.readouterr().out)["tools"]["score"]["state"] == "stopped"
