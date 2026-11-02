"""A resume must preserve its budget and must not reapply warm initialization."""

import subprocess

import pytest

from scripts.monitor_training import with_resume, stop_grace_seconds, validate_source


def test_resume_replaces_initialization_without_resetting_time_cap():
    command = ["python", "train.py", "--max-wall-seconds", "40775.07150149765",
               "--world-model-from", "physics.pt", "--initialize-from", "legacy.pt",
               "--resume-budget-record", "one-time-authorization.json",
               "--resume", "stale.pt", "--resume-runtime-seconds-floor", "1"]
    resumed = with_resume(command, "model_000500.pt", 700.5)
    assert "--world-model-from" not in resumed
    assert "--initialize-from" not in resumed
    assert "--resume-budget-record" not in resumed
    assert resumed.count("--resume") == 1
    assert resumed[resumed.index("--resume") + 1] == "model_000500.pt"
    assert float(resumed[resumed.index("--resume-runtime-seconds-floor") + 1]) == 700.5
    assert resumed[resumed.index("--max-wall-seconds") + 1] == "40775.07150149765"
    assert command[command.index("--resume") + 1] == "stale.pt"


def test_stop_grace_cannot_extend_four_hour_budget():
    assert stop_grace_seconds(14400, 14220) == 180
    assert stop_grace_seconds(14400, 14225) == 175
    assert stop_grace_seconds(14400, 14400) == 0
    assert stop_grace_seconds(14400, 14405) == 0


def test_source_contract_rejects_dirty_source_and_changed_head_but_allows_outputs(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    (tmp_path/"train.py").write_text("version = 1\n")
    (tmp_path/".gitignore").write_text("outputs/\n.reference/\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "source")
    cfg = {"source_commit": git("rev-parse", "HEAD")}
    validate_source(cfg, tmp_path)
    (tmp_path/"outputs").mkdir()
    (tmp_path/"outputs/metrics.csv").write_text("iteration,reward\n")
    validate_source(cfg, tmp_path)
    (tmp_path/"train.py").write_text("version = 2\n")
    with pytest.raises(ValueError, match="tracked changes"):
        validate_source(cfg, tmp_path)
    validate_source({}, tmp_path)  # Legacy campaign configs retain their contract.
    git("add", "train.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "new source")
    with pytest.raises(ValueError, match="HEAD"):
        validate_source(cfg, tmp_path)
