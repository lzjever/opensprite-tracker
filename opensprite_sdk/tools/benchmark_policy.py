#!/usr/bin/env python3
"""Measure CPU actor inference on this host; never opens robot devices."""
import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
from opensprite_sdk.policy import PolicyBundle


def benchmark(path, iterations=1000, warmup=100):
    if not 1<=iterations<=100000 or not 0<=warmup<=10000:
        raise ValueError("invalid benchmark sample count")
    bundle=PolicyBundle(path)
    inputs={"observation_history":np.zeros((1,10,99),np.float32),
            "goal":bundle.reference["goals"][:1].astype(np.float32)}
    elapsed=[]
    for index in range(warmup+iterations):
        start=time.perf_counter_ns()
        output=bundle.session.run(["action"],inputs)[0]
        dt=(time.perf_counter_ns()-start)/1e6
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite inference")
        if index>=warmup:
            elapsed.append(dt)
    return {"architecture":platform.machine(),"provider":bundle.session.get_providers(),
            "samples":iterations,"warmup":warmup,"batch_size":1,
            "actor_only_ms":dict(zip(("p50","p95","p99","max"),map(float, np.percentile(elapsed,[50,95,99,100])))),
            "exceeded_20ms":sum(t>20 for t in elapsed),
            "includes_sensor_or_can_latency":False,"hardware_opened":False}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path)
    p.add_argument("--iterations",type=int,default=1000)
    p.add_argument("--warmup",type=int,default=100)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=benchmark(args.bundle,args.iterations,args.warmup)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
