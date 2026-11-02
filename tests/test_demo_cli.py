"""CPU-only command planning and missing-LFS guards; never launch the simulator."""
from pathlib import Path
import json
import subprocess
import sys
import venv

import pytest

from scripts import demo
from scripts.check_install import file_problem, pinned_versions


def parse(*args):
    return demo.parser().parse_args(args)


def test_record_replays_18k_and_owned_assets_without_old_metadata_path_binding():
    plan = demo.build_plan(parse("record", "--dry-run"))
    evaluate, render = plan["commands"]
    assert "--continue-after-failure" in evaluate and "--max-steps" not in evaluate
    assert Path(evaluate[evaluate.index("--checkpoint")+1]).name == "model_018000.pt"
    assert evaluate[evaluate.index("--num-envs")+1] == "1"
    assert render[render.index("--model")+1] == str(demo.ROOT/"assets/opensprite/opensprite.xml")
    assert "--checkpoint" not in render and "--include-post-failure" in render
    assert render[render.index("--width")+1] == "1920"
    assert render[render.index("--fps")+1] == "50"


def test_training_plan_is_fresh_8h_and_does_not_write_active_config():
    active = demo.ROOT/"configs/training.json"
    before = active.read_bytes() if active.exists() else None
    first = demo.build_plan(parse("train", "--dry-run"))
    second = demo.build_plan(parse("train", "--dry-run"))
    assert first["create_parent"] != second["create_parent"]
    assert not Path(first["create_parent"]).exists()
    cfg = first["config"]
    assert (cfg["budget_seconds"], cfg["hard_budget_seconds"], cfg["prior_training_seconds"]) == (28620, 28800, 0)
    assert cfg["run_dir"].startswith("outputs/demo/train-dance-")
    assert "goal-sim-" in cfg["run_dir"]
    assert cfg["source_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=demo.ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    assert not any(flag in cfg["command"] for flag in ("--resume", "--initialize-from", "--world-model-from", "--resume-budget-record"))
    assert first["config_path"].endswith("demo-training.json")
    assert (active.read_bytes() if active.exists() else None) == before
    with pytest.raises(ValueError, match="already exists"):
        demo.build_plan(parse("train", "--output", "outputs"))


def test_sim2real_preset_preserves_recipe_and_gets_independent_cold_ledger():
    old = json.loads((demo.ROOT/"configs/demo/dance-8h.json").read_text())
    new = json.loads((demo.ROOT/"configs/demo/dance-goal-sim2real-8h.json").read_text())
    assert new["command"] == old["command"] + ["--training-variant", "goal-sim2real"]
    legacy = demo.build_plan(parse("train", "--dry-run"))
    plan = demo.build_plan(parse("train", "--preset", "goal-sim2real", "--dry-run"))
    cfg = plan["config"]
    assert "train-dance-goal-sim2real-" in cfg["run_dir"]
    assert cfg["run_dir"] != legacy["config"]["run_dir"]
    assert cfg["delivery_dir"] == cfg["run_dir"] + "/delivery"
    assert cfg["delivery_dir"] != legacy["config"]["delivery_dir"]
    assert (cfg["budget_seconds"], cfg["hard_budget_seconds"], cfg["prior_training_seconds"]) == (28620, 28800, 0)
    assert cfg["evaluation_clips"] == [0] and cfg["evaluation_seeds"] == [42, 43, 44]
    assert cfg["delivery_video_count"] == 0
    assert cfg["command"][-2:] == ["--training-variant", "goal-sim2real"]
    assert not Path(plan["create_parent"]).exists()


def test_explicit_paths_are_single_argv_tokens_and_do_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plan = demo.build_plan(parse("evaluate", "--checkpoint", "artifacts/a checkpoint.pt", "--motion", "data/a motion.npz"))
    argv = plan["commands"][0]
    assert argv[argv.index("--checkpoint")+1] == str(demo.ROOT/"artifacts/a checkpoint.pt")
    assert argv[argv.index("--motion")+1] == str(demo.ROOT/"data/a motion.npz")
    assert argv[argv.index("--num-envs")+1] == "3"
    assert plan["cwd"] == str(demo.ROOT)
    with pytest.raises(ValueError, match="repository-owned"):
        demo.build_plan(parse("evaluate", "--motion", ".reference/source.npz"))


def test_retarget_and_build_motion_forward_single_source_and_cpu_default():
    native = demo.build_plan(parse("retarget", "--stage", "all", "--motions", "dance2_subject1", "--correspondence-dir", "outputs/fresh-correspondence"))["commands"][0]
    assert native[native.index("--motions")+1] == "dance2_subject1"
    assert native[native.index("--correspondence-device")+1] == "cpu"
    assert native[native.index("--correspondence-dir")+1] == str(demo.ROOT/"outputs/fresh-correspondence")
    converted = demo.build_plan(parse("build-motion"))["commands"][0]
    assert converted[converted.index("--stage")+1] == "convert"
    assert "--motion-output-dir" in converted


def test_missing_default_umr_environment_never_falls_back_to_training_python(monkeypatch):
    monkeypatch.setattr(demo, "python_path", lambda *a: "training-python")
    command = demo.build_plan(parse("retarget"))["commands"][0]
    assert command[0] == str(demo.ROOT/".venv-umr/bin/python")


def test_lfs_pointer_stops_before_interpreter_or_subprocess(tmp_path, monkeypatch):
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 42\n")
    assert "Git LFS pointer" in file_problem(checkpoint)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Must not dispatch a process for a pointer"))
    with pytest.raises(ValueError, match="git lfs"):
        demo.execute({"required_files": [str(checkpoint)], "commands": [["missing-python", "train"]]}, "evaluate")


def test_dry_run_does_not_execute_or_create_a_run(monkeypatch, capsys):
    monkeypatch.setattr(demo, "execute", lambda *a: pytest.fail("dry-run must not execute"))
    demo.main(["train", "--dry-run"])
    assert '"prior_training_seconds": 0' in capsys.readouterr().out


def test_pin_reader_keeps_cuda_versions_and_skips_index_or_git_urls(tmp_path):
    path = tmp_path/"requirements.txt"
    path.write_text("--extra-index-url https://example.test\ntorch==2.8.0+cu128\nNumPy==1.26.4\nsmplx @ git+https://example.test/repo@abc\n")
    assert pinned_versions(path) == {"torch": "2.8.0+cu128", "numpy": "1.26.4"}


def test_python_symlink_preserves_virtual_environment_for_explicit_paths(tmp_path, monkeypatch):
    from scripts import check_install

    environment = tmp_path/"isolated-venv"
    venv.EnvBuilder(symlinks=True, with_pip=False).create(environment)
    executable = environment/"bin/python"
    assert executable.is_symlink()
    monkeypatch.setattr(check_install, "ROOT", tmp_path)
    for value in (str(executable), "isolated-venv/bin/python"):
        selected = check_install.python_path(value, Path(sys.executable))
        assert selected == str(executable)
        result = subprocess.run([selected, "-c", "import sys; print(sys.prefix)"],
                                check=True, text=True, capture_output=True)
        assert result.stdout.strip() == str(environment)
