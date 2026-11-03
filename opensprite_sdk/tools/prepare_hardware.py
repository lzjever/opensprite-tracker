#!/usr/bin/env python3
"""Generate native inputs and print a launch argv; never opens hardware."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from opensprite_sdk.backends.hardware_config import HardwareConfig

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hardware',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--actuate',action='store_true')
p.add_argument('--acknowledge-actuation',action='store_true')
a=p.parse_args()
config=HardwareConfig.load(a.hardware)
command=config.launch_command(a.output,actuate=a.actuate,acknowledge=a.acknowledge_actuation)
print(json.dumps({'opens_hardware':False,'native_argv':command},indent=2))
