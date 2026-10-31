"""Local compilation caches and the system CUDA driver's compiler link name."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_runtime():
    for key, folder in (
        ("WARP_CACHE_PATH", "warp"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        os.environ.setdefault(key, str(ROOT / ".cache" / folder))
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    for directory in (
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
    ):
        driver = directory / "libcuda.so.1"
        if not driver.exists():
            continue
        if (directory / "libcuda.so").exists():
            return
        target = ROOT / ".cache/cuda-link"
        target.mkdir(parents=True, exist_ok=True)
        link = target / "libcuda.so"
        if not link.exists():
            link.symlink_to(driver.resolve())
        os.environ["TRITON_LIBCUDA_PATH"] = str(target)
        return
