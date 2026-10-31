#!/usr/bin/env python3
"""Read training telemetry without touching the GPU or loading checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess


def snapshot(run_dir):
    run = Path(run_dir)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run)}
    checkpoints = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        path = checkpoints[-1]
        result["checkpoint"] = {"path": str(path), "iteration": int(path.stem.split("_")[-1]),
                                "bytes": path.stat().st_size}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_dir = run / "tensorboard" if (run / "tensorboard").is_dir() else run
        result["event_dir"] = str(event_dir)
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 2})
        accumulator.Reload()
        metrics = {}
        for tag in accumulator.Tags().get("scalars", []):
            if not tag.startswith(("clock/", "real/", "model/", "sampling/", "sampler/",
                                   "memory/", "lr/", "time/", "actor/", "critic/", "clip",
                                   "imagination/", "loss/", "timing/", "perf/", "physics/")):
                continue
            events = accumulator.Scalars(tag)
            if events:
                event = events[-1]
                metrics[tag] = {"step": event.step, "value": event.value,
                                "finite": math.isfinite(event.value)}
        result["metrics"] = metrics
        result["nonfinite_metrics"] = [k for k, v in metrics.items() if not v["finite"]]
    except Exception as error:
        result["telemetry_error"] = str(error)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    result["gpu"] = gpu.stdout.strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    value = json.dumps(snapshot(args.run_dir), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(value + "\n")
    print(value)
