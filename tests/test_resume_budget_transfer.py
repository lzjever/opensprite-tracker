"""Authorized clock transfer is separate from loading or ordinary resume."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from opensprite_tracker.supermimic.train import (
    parser, resume_budget_transfer, resume_clock_offset, write_resume_transfer_proof,
    save_transfer_checkpoint,
)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    source = old / "model_010000.pt"
    source.touch()
    completion = old / "completion.json"
    completion.write_text(json.dumps({"iteration": 10000, "stop_reason": "wall_budget"}))
    supervisor = old / "supervisor.json"
    supervisor.write_text(json.dumps({"status": "finished", "returncode": 0,
                                      "active_seconds": 14222., "pid": 999999}))
    def no_live_old_process(pid, signal):
        assert pid == 999999 and signal == 0
        raise ProcessLookupError()
    monkeypatch.setattr("opensprite_tracker.supermimic.train.os.kill", no_live_old_process)
    ck = {"args": {"output": str(old), "wm_only_iters": 500, "max_wall_seconds": 14220.},
          "wall_seconds": 14221.75, "iteration": 10000, "sim_steps": 19200000,
          "sampling": {"returns": torch.ones(2), "past": torch.zeros(2)},
          "initialization": {"mode": "cold_start"}}
    record = {"schema": 1, "mode": "authorized_extension", "source_checkpoint": str(source),
              "old_output": str(old), "new_output": str(new), "source_iteration": 10000,
              "old_runtime_seconds": ck["wall_seconds"], "new_hard_budget_seconds": 14400,
              "new_soft_budget_seconds": 14220, "user_authorization": "完成后再加四个小时继续训练",
              "source_completion": str(completion), "source_supervisor": str(supervisor),
              "source_completion_reason": "wall_budget", "parent_budget_seconds": 14220.}
    record_path = tmp_path / "authorization.json"
    record_path.write_text(json.dumps(record))
    args = SimpleNamespace(resume_budget_record=record_path, resume=source,
                           initialize_from=None, world_model_from=None, output=new,
                           max_wall_seconds=14220., resume_runtime_seconds_floor=0.,
                           wm_only_iters=500)
    return ck, args, record


def test_transfer_preserves_absolute_progress_and_separates_old_clock(campaign):
    ck, args, _ = campaign
    ancestry = resume_budget_transfer(ck, args)
    assert ancestry["source_iteration"] == ck["iteration"] == 10000
    assert ancestry["old_runtime_seconds"] == 14221.75
    assert ancestry["source_initialization"] == {"mode": "cold_start"}
    assert resume_clock_offset(ck, args, ancestry) == 0.
    # Internal resumes omit the one-time record and retain their OWN new clock.
    args.resume_budget_record = None
    args.resume_runtime_seconds_floor = 120.
    saved = {"wall_seconds": 100., "budget_ancestry": ancestry}
    assert resume_budget_transfer(saved, args) is None
    assert resume_clock_offset(saved, args, None) == 120.
    saved["wall_seconds"] = 150.
    assert resume_clock_offset(saved, args, None) == 150.


@pytest.mark.parametrize("key,value", [
    ("source_iteration", 9999), ("old_runtime_seconds", 0.),
    ("new_hard_budget_seconds", 28800), ("new_soft_budget_seconds", 14400),
    ("user_authorization", ""), ("source_checkpoint", "relative.pt"),
    ("source_completion", "/wrong/completion.json"),
])
def test_mismatched_or_missing_authorization_is_rejected(campaign, key, value):
    ck, args, record = campaign
    record[key] = value
    args.resume_budget_record.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        resume_budget_transfer(ck, args)


@pytest.mark.parametrize("repeat_kind", ["existing_checkpoint", "existing_proof", "ancestry"])
def test_authorization_cannot_reset_an_existing_extension_clock(campaign, repeat_kind):
    ck, args, _ = campaign
    if repeat_kind == "existing_checkpoint":
        (args.output / "model_010001.pt").touch()
    elif repeat_kind == "existing_proof":
        (args.output / "resume-transfer-proof.json").touch()
    else:
        ck["budget_ancestry"] = {"mode": "authorized_budget_extension"}
    with pytest.raises(ValueError, match="once"):
        resume_budget_transfer(ck, args)


def test_unfinished_run_and_reapplied_warmup_are_rejected(campaign):
    ck, args, record = campaign
    completion = args.resume.parent / "completion.json"
    completion.write_text(json.dumps({"iteration": 10000, "stop_reason": "signal"}))
    # Merely replacing the completion reason without matching the record fails.
    with pytest.raises(ValueError, match="finish its wall budget"):
        resume_budget_transfer(ck, args)
    completion.write_text(json.dumps({"iteration": 10000, "stop_reason": "wall_budget"}))
    args.wm_only_iters = 10000
    with pytest.raises(ValueError, match="completed WM warmup"):
        resume_budget_transfer(ck, args)


def test_budget_completed_signal_is_allowed_but_early_manual_signal_is_not(campaign):
    ck, args, record = campaign
    completion = args.resume.parent / "completion.json"
    completion.write_text(json.dumps({"iteration": 10000, "stop_reason": "signal"}))
    record["source_completion_reason"] = "signal"
    args.resume_budget_record.write_text(json.dumps(record))
    assert resume_budget_transfer(ck, args)["source_completion_reason"] == "signal"
    supervisor_path = args.resume.parent / "supervisor.json"
    saved = json.loads(supervisor_path.read_text())
    for invalid in ({"status": "stopped"}, {"active_seconds": 14219.}, {"returncode": 1}):
        supervisor_path.write_text(json.dumps({**saved, **invalid}))
        with pytest.raises(ValueError, match="Source supervisor"):
            resume_budget_transfer(ck, args)


def test_baseline_checkpoint_makes_pre_refill_failure_recoverable_without_reusing_record(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path), "--disable-torque-lag"])
    learner = SimpleNamespace(
        initialization={"mode": "full_training_resume_with_authorized_budget_extension"},
        budget_ancestry={"old_runtime_seconds": 14221.75},
        mdp=SimpleNamespace(cfg=SimpleNamespace(profile="legacy_full")),
    )
    for name, opt_name in (("wm", "opt_wm"), ("pi", "opt_pi"),
                           ("critic", "opt_v"), ("transition_model", "opt_transition")):
        model = torch.nn.Linear(2, 1)
        setattr(learner, name, model)
        setattr(learner, opt_name, torch.optim.Adam(model.parameters()))
    sampler = SimpleNamespace(returns=torch.ones(2), past=torch.zeros(2))
    path = save_transfer_checkpoint(learner, args, 10000, 19200000, 23.5, sampler)
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert path.name == "model_010000.pt" and not path.with_suffix(".pt.tmp").exists()
    assert saved["iteration"] == 10000 and saved["sim_steps"] == 19200000
    assert saved["wall_seconds"] == 23.5
    assert saved["budget_ancestry"] == learner.budget_ancestry
    assert saved["initialization"]["mode"] != "cold_start"
    assert torch.equal(saved["sampling"]["returns"], sampler.returns)
    args.resume = path
    args.resume_runtime_seconds_floor = 30.
    assert resume_budget_transfer(saved, args) is None
    assert resume_clock_offset(saved, args, None) == 30.


def test_exact_proof_checks_optimizer_moments_normalizers_and_sampler_before_refill(tmp_path):
    learner = SimpleNamespace()
    for name in ("wm", "pi", "critic", "transition_model"):
        model = torch.nn.Linear(2, 1)
        model.register_buffer("audit_count", torch.tensor(23.))
        setattr(learner, name, model)
        opt_name = {"wm": "opt_wm", "pi": "opt_pi", "critic": "opt_v",
                    "transition_model": "opt_transition"}[name]
        optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        setattr(learner, opt_name, optimizer)
    names = ("wm", "pi", "critic", "transition_model", "opt_wm", "opt_pi", "opt_v", "opt_transition")
    ck = {name: deepcopy(getattr(learner, name).state_dict()) for name in names}
    sampler = SimpleNamespace(returns=torch.ones(2), past=torch.arange(2))
    ck.update(sampling={"returns": sampler.returns.clone(), "past": sampler.past.clone()},
              iteration=10000, sim_steps=19200000)
    path = tmp_path / "proof.json"
    proof = write_resume_transfer_proof(learner, sampler, ck, {"mode": "extension"}, path)
    assert proof["pass"] and all(proof["exact_state_equal"].values())
    assert proof["absolute_iteration"] == 10000 and proof["new_runtime_offset_seconds"] == 0
    assert proof["optimizer_steps"] == {name: [1] for name in names if name.startswith("opt_")}
    assert proof["warmup_reapplied"] is False
    with pytest.raises(FileExistsError):
        write_resume_transfer_proof(learner, sampler, ck, {}, path)
    moment = next(iter(learner.opt_wm.state.values()))["exp_avg"]
    moment.add_(1.)
    with pytest.raises(ValueError, match="all loaded state"):
        write_resume_transfer_proof(learner, sampler, ck, {}, tmp_path / "bad-proof.json")
    assert not (tmp_path / "bad-proof.json").exists()
