"""Exercise the background launcher with fake Codex, never the real trainer/API."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hourly_audit.py"
spec = importlib.util.spec_from_file_location("hourly_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def campaign(tmp_path):
    run = tmp_path / "outputs/train/test"
    run.mkdir(parents=True)
    (tmp_path / "task.md").write_text("Read actual training metrics, preserve 43200 second budget.")
    (tmp_path / "config.json").write_text(json.dumps({
        "run_dir": "outputs/train/test", "budget_seconds": 43020,
        "prior_training_seconds": 1031.571499087017,
    }))
    (run / "supervisor.json").write_text(json.dumps({"active_seconds": 3100, "status": "completed"}))
    fake = tmp_path / "fake-codex"
    fake.write_text("""#!/usr/bin/env python3
import json,os,pathlib,sys,time
root=pathlib.Path(os.environ['FAKE_ROOT'])
with (root/'calls.jsonl').open('a') as stream:stream.write(json.dumps(sys.argv[1:])+'\\n')
prompt=sys.stdin.read()
(root/'prompt.txt').write_text(prompt)
time.sleep(float(os.environ.get('FAKE_DELAY','0')))
if os.environ.get('FAKE_FAIL') == '1':sys.exit(7)
pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('逐动作趋势审计完成。')
""")
    fake.chmod(0o755)
    env = dict(os.environ, FAKE_ROOT=str(tmp_path))
    command = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--config", "config.json",
               "--task", "task.md", "--codex", str(fake)]
    return tmp_path, run, command, env


@pytest.mark.parametrize("failed", [False, True])
def test_single_audit_success_or_failure_preserves_training_and_budget(campaign, failed):
    root, run, command, env = campaign
    if failed:
        env["FAKE_FAIL"] = "1"
        (run / "hourly").mkdir()
        (run / "hourly/latest.md").write_text("Previous successful report")
    original = (run / "supervisor.json").read_bytes()
    result = subprocess.run(command + ["--once"], env=env, capture_output=True, timeout=10)
    assert result.returncode == (1 if failed else 0)
    state = json.loads((run / "hourly/runner-state.json").read_text())
    assert state["status"] == ("error" if failed else "completed")
    assert (run / "supervisor.json").read_bytes() == original
    calls = [json.loads(x) for x in (root / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0][:2] == ["-a", "never"]
    assert 'approval_policy="never"' in calls[0]
    assert 'sandbox_mode="danger-full-access"' in calls[0]
    assert not any(x in calls[0] for x in ("--model", "-m", "--ignore-rules", "--dangerously-bypass-hook-trust"))
    latest = (run / "hourly/latest.md").read_text()
    assert latest == ("Previous successful report" if failed else "逐动作趋势审计完成。")
    snapshot = json.loads(next((run / "hourly").glob("*.snapshot.json")).read_text())
    assert snapshot["campaign_seconds"] == 3100
    assert snapshot["formal_elapsed_seconds"] == pytest.approx(2068.428500912983)


def test_lifetime_lock_prevents_second_launcher_even_between_runs(campaign):
    root, run, command, env = campaign
    process = subprocess.Popen(command + ["--interval-seconds", "30"], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            state_file = run / "hourly/runner-state.json"
            if state_file.is_file() and json.loads(state_file.read_text()).get("launcher_status") == "waiting":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Fake first audit did not finish")
        result = subprocess.run(command + ["--once"], env=env, capture_output=True, timeout=5)
        assert result.returncode == 0 and b"owns this campaign" in result.stdout
        assert len((root / "calls.jsonl").read_text().splitlines()) == 1
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_delivery_requires_full_requests_shared_model_and_local_files(campaign):
    root, run, _, _ = campaign
    delivery = root / audit.DELIVERY
    delivery.mkdir(parents=True)
    (run / "hourly").mkdir()
    for name in ("checkpoint.pt", "README.md", "clip0.mp4", "clip1.mp4", "clip2.mp4"):
        (delivery / name).write_bytes(b"actual artifact fixture")
    reports = []
    for clip in range(3):
        for seed in (42, 43, 44):
            path = delivery / f"clip{clip}-seed{seed}.json"
            path.write_text(json.dumps({"clip": clip, "seed": seed, "checkpoint": str(delivery / "checkpoint.pt"),
                "num_envs": 3, "expected_transitions": 1000, "requested_transitions": 1000,
                "executed_transitions": 50, "complete_clip": False, "success_rate": 0}))
            reports.append(str(path))
    marker = {"status": "complete", "quality_pass": False, "quality_notes": "Failed early honestly reported",
              "completed_at": "2026-11-01T00:00:00Z", "delivery_dir": str(delivery),
              "checkpoint": str(delivery / "checkpoint.pt"), "readme": str(delivery / "README.md"),
              "evaluation_reports": reports, "videos": [str(delivery / f"clip{i}.mp4") for i in range(3)]}
    (run / "hourly/delivery-complete.json").write_text(json.dumps(marker))
    assert audit.delivery_complete(root, run)  # Completion can honestly fail quality.
    report = Path(reports[0])
    original = report.read_text()
    value = json.loads(original)
    value["requested_transitions"] = 500
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prefix"):
        audit.delivery_complete(root, run)
    value["requested_transitions"] = 1000
    value["checkpoint"] = str(delivery / "another.pt")
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="same shared"):
        audit.delivery_complete(root, run)
    report.write_text(original)
    (delivery / "clip0.mp4").unlink()
    (delivery / "clip0.mp4").symlink_to(delivery / "clip1.mp4")
    with pytest.raises(ValueError, match="symlink"):
        audit.delivery_complete(root, run)
