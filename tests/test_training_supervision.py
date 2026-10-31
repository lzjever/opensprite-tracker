"""A resume must preserve its budget and must not reapply warm initialization."""

from scripts.monitor_training import with_resume


def test_resume_replaces_initialization_without_resetting_time_cap():
    command = ["python", "train.py", "--max-wall-seconds", "40775.07150149765",
               "--world-model-from", "physics.pt", "--initialize-from", "legacy.pt",
               "--resume", "stale.pt", "--resume-runtime-seconds-floor", "1"]
    resumed = with_resume(command, "model_000500.pt", 700.5)
    assert "--world-model-from" not in resumed
    assert "--initialize-from" not in resumed
    assert resumed.count("--resume") == 1
    assert resumed[resumed.index("--resume") + 1] == "model_000500.pt"
    assert float(resumed[resumed.index("--resume-runtime-seconds-floor") + 1]) == 700.5
    assert resumed[resumed.index("--max-wall-seconds") + 1] == "40775.07150149765"
    assert command[command.index("--resume") + 1] == "stale.pt"
