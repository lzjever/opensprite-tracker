#!/usr/bin/env python3
"""Read-only CPU installation checks. Never loads a policy or creates a GPU context."""
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = Path("data/motions/lafan_dance.npz")
DEFAULT_CHECKPOINT = Path("artifacts/checkpoints/model_030000.pt")
TRAIN_PACKAGES = ("torch", "numpy", "scipy", "mujoco", "mujoco-warp", "mjlab",
                  "warp-lang", "tensorboard", "osqp", "imageio", "imageio-ffmpeg", "pillow")
UMR_PACKAGES = ("torch", "numpy", "scipy", "mujoco", "py-soma-x", "clarabel",
                "cholespy", "trimesh", "embreex", "rtree", "smplx", "osqp")


def repo_path(value, root=ROOT):
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (root/value).resolve()


def is_lfs_pointer(path):
    with Path(path).open("rb") as stream:
        return stream.read(200).startswith(b"version https://git-lfs.github.com/spec/v1")


def file_problem(path):
    path = Path(path)
    if not path.is_file():
        return f"Missing file: {path}. Pull the repository's assets with git lfs pull."
    if path.stat().st_size == 0:
        return f"Empty file: {path}"
    if is_lfs_pointer(path):
        return f"Git LFS pointer instead of file contents: {path}. Run git lfs install && git lfs pull."
    return None


def python_path(value, fallback):
    if value:
        if "/" in str(value):
            # Python locates pyvenv.cfg from the invoked executable path.
            # Resolving the final symlink would silently select the base env.
            path = Path(value).expanduser()
            return os.path.abspath(path if path.is_absolute() else ROOT/path)
        return shutil.which(str(value)) or str(value)
    return str(fallback if fallback.is_file() else Path(sys.executable))


def pinned_versions(path):
    versions = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+)(?:\[[^]]+\])?==([^\s;]+)", line)
        if match:
            versions[match[1].lower().replace("_", "-")] = match[2]
    return versions


def cpu_command(command, cwd=ROOT):
    return subprocess.run([str(x) for x in command], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def inspect_python(executable, packages, lock, add):
    probe = ("import importlib.metadata as m,json,sys; v={}; "
             "exec('for n in sys.argv[1:]:\\n try: v[n]=m.version(n)\\n except m.PackageNotFoundError: v[n]=None'); "
             "print(json.dumps({'python':list(sys.version_info[:3]),'versions':v}))")
    try:
        result = cpu_command([executable, "-c", probe, *packages])
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        found = json.loads(result.stdout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        add("fail", f"python:{executable}", f"Cannot inspect environment: {error}. Create its Python3.12 venv and install {lock.name}.")
        return
    add("pass" if found["python"][:2] == [3, 12] else "fail", f"python:{executable}", str(found["python"]))
    pinned = pinned_versions(lock)
    for name, version in found["versions"].items():
        expected = pinned.get(name)
        good = version is not None and (expected is None or version == expected)
        add("pass" if good else "fail", f"package:{name}",
            f"{version or 'not installed'}; expected {expected or 'installed from '+lock.name}")


def check(args):
    checks = []
    def add(status, name, detail):
        checks.append({"status": status, "name": name, "detail": str(detail)})
    def asset(path, required=True):
        path = repo_path(path, root=ROOT)
        if not path.is_relative_to(ROOT):
            add("fail", str(path), "Runtime asset resolves outside the repository; copy owned assets here rather than using a sibling checkout/symlink.")
            return False
        problem = file_problem(path)
        add("fail" if required and problem else "warn" if problem else "pass", str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), problem or "Present, nonempty, not a Git LFS pointer")
        return problem is None
    for path in ("assets/opensprite/opensprite.xml", "assets/opensprite/retarget_scene.xml", "assets/geno/Geno.fbx",
                 "data/sources/lafan1-resolved/dance2_subject1.bvh",
                 "data/sources/lafan1-resolved/Geno_stance.bvh",
                 "assets/umr_correspondence/geno_opensprite0825/dataset.npz",
                 "assets/umr_correspondence/geno_opensprite0825/training/correspondence_slots_final.npz"):
        asset(path)
    model = ROOT/"assets/opensprite/opensprite.xml"
    if file_problem(model) is None:
        try:
            xml = ET.parse(model).getroot()
            compiler = xml.find("compiler")
            meshdir = compiler.get("meshdir", "") if compiler is not None else ""
            for mesh in xml.findall("./asset/mesh"):
                asset(model.parent/meshdir/mesh.attrib["file"])
        except (OSError, ET.ParseError, KeyError) as error:
            add("fail", "model XML", error)
    if args.profile in ("demo", "all"):
        inspect_python(python_path(args.python, ROOT/".venv/bin/python"), TRAIN_PACKAGES, ROOT/"requirements-lock.txt", add)
        asset(args.motion)
        asset(args.checkpoint)
        asset("artifacts/videos/dance-model018000-full-1080p50.mp4", required=False)
        for tool in ("ffmpeg", "ffprobe"):
            exe = shutil.which(tool)
            if exe:
                result = cpu_command([exe, "-version"])
                add("pass" if result.returncode == 0 else "fail", tool, result.stdout.splitlines()[0] if result.stdout else result.stderr)
            else:
                add("fail", tool, "Install system FFmpeg (both ffmpeg and ffprobe are needed for recording/verification).")
        egl = ctypes.util.find_library("EGL")
        add("pass" if egl else "fail", "EGL library", egl or "Install the EGL runtime and compatible GPU driver for headless rendering.")
        add("warn", "GPU execution", "Not exercised: this doctor creates no CUDA/EGL context. GPU simulation/rendering readiness is not proven by these CPU checks.")
    if args.profile in ("retarget", "all"):
        umr_python = python_path(args.umr_python, ROOT/".venv-umr/bin/python")
        if not args.umr_python and not (ROOT/".venv-umr/bin/python").is_file():
            add("fail", "UMR environment", "Create .venv-umr separately: its NumPy/MuJoCo versions differ from the training venv.")
        else:
            inspect_python(umr_python, UMR_PACKAGES, ROOT/"requirements-umr.txt", add)
        for path in ("third_party/umr/scripts/package.json", "third_party/umr/scripts/package-lock.json",
                     "third_party/umr/scripts/FBXLoaderPatched.js", "third_party/umr/scripts/nr_three_loader.mjs",
                     "third_party/umr/scripts/vendor/three/build/three.module.js"):
            asset(path)
        node = shutil.which("node")
        if not node:
            add("fail", "Node.js", "Install Node.js >=18. Vendored Three.js0.165.0 is included; no global npm package is required.")
        else:
            result = cpu_command([node, "--version"])
            match = re.match(r"v(\d+)", result.stdout.strip())
            supported = result.returncode == 0 and match and int(match[1]) >= 18
            add("pass" if supported else "fail", "Node.js", f"{result.stdout.strip()}; minimum18 from third_party/umr/scripts/package.json")
            if supported:
                result = cpu_command([node, "--no-warnings", "--experimental-loader", "./nr_three_loader.mjs",
                    "--input-type=module", "-e", "Promise.all([import('three'),import('./FBXLoaderPatched.js')]).then(([m])=>console.log('Three.js revision '+m.REVISION+' and FBX parser ready'))"], cwd=ROOT/"third_party/umr/scripts")
                add("pass" if result.returncode == 0 else "fail", "vendored Node runtime", result.stdout.strip() or result.stderr.strip())
    git = shutil.which("git")
    result = cpu_command([git, "lfs", "version"]) if git else None
    add("pass" if result and result.returncode == 0 else "warn", "Git LFS", result.stdout.strip() if result and result.returncode == 0 else "Install Git LFS and run git lfs pull if large files are pointers or missing.")
    return {"root": str(ROOT), "profile": args.profile, "passed": not any(c["status"] == "fail" for c in checks), "checks": checks,
            "scope": "Read-only CPU metadata/files/tool checks; no install, checkpoint load, simulation, training, CUDA or rendering."}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("demo", "retarget", "all"), default="demo")
    p.add_argument("--python", help="training interpreter; default .venv/bin/python")
    p.add_argument("--umr-python", help="retarget interpreter; default .venv-umr/bin/python")
    p.add_argument("--motion", default=str(DEFAULT_MOTION))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--json", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = check(args)
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        report = {"root": str(ROOT), "profile": args.profile, "passed": False,
                  "checks": [{"status": "fail", "name": "installation probe", "detail": str(error)}]}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report["checks"]:
            print(f"[{item['status'].upper()}] {item['name']}: {item['detail']}")
        print("CPU checks passed; GPU execution was not tested." if report["passed"] else "Installation checks failed; fix the listed items before running the demo.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
