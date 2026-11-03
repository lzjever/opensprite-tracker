#!/usr/bin/env python3
"""Verify bundle hashes and CPU ONNX/PyTorch export parity."""
import argparse
import json
from pathlib import Path
import numpy as np
from opensprite_sdk.policy import PolicyBundle


def check(path):
    bundle = PolicyBundle(path)
    with np.load(Path(path)/"parity.npz", allow_pickle=False) as data:
        actual = bundle.session.run(["action"], {k:data[k] for k in ("observation_history", "goal")})[0]
        np.testing.assert_allclose(actual, data["action"], rtol=1e-4, atol=1e-4)
        error = float(np.max(np.abs(actual-data["action"])))
    return {"hashes_verified":True, "onnx_parity_max_abs":error, "frames":bundle.contract["frames"],
            "provider":bundle.session.get_providers(), "hardware_qualified":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    result=check(args.bundle)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
