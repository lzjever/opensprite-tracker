#!/usr/bin/env python3
"""Print a small current-run status without loading checkpoints or using CUDA."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', help='defaults to configs/training.json run_dir')
args = parser.parse_args()
run = Path(args.run_dir or json.loads(Path("configs/training.json").read_text())["run_dir"])
state = json.loads((run / 'supervisor.json').read_text())
result = {k: state.get(k) for k in ('status', 'pid', 'restarts', 'returncode')}
result['campaign_hours'] = round(state.get('active_seconds', 0) / 3600, 4)
result['utc'] = state.get('utc')
logs = sorted(run.glob('train-segment-*.log'))
if logs:
    with logs[-1].open('rb') as stream:
        stream.seek(max(0, logs[-1].stat().st_size - 131072))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        if line.startswith('UPDATE '):
            _, iteration, payload = line.split(' ', 2)
            metrics = json.loads(payload)
            result['iteration'] = int(iteration)
            keep = ('real/reward', 'real/episode_length_steps', 'real/terminated_per_step',
                    'model/loss', 'model/error_per_valid_transition', 'model/last_step_valid_fraction',
                    'time/iterations_per_second', 'memory/peak_gib')
            result['metrics'] = {k: round(float(v), 5) for k, v in metrics.items()
                                 if k in keep or k.startswith('clip') or 'grad_norm' in k}
            break
print(json.dumps(result, ensure_ascii=False, indent=2))
