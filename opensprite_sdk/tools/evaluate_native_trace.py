#!/usr/bin/env python3
"""Score isolated simulator diagnostics; never supply truth to the actor."""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation(q):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def evaluate(truth,reference):
    with np.load(reference,allow_pickle=False) as data:
        if 'qpos' in data:
            positions=data['qpos'][:,:3];quaternions=data['qpos'][:,3:7]
        else:
            root=data['body_names'].astype(str).tolist().index('pelvis_link')
            positions=data['body_pos_w'][:,root];quaternions=data['body_quat_w'][:,root]
    last=0;first=None;height_max=gravity_max=0.
    with Path(truth).open() as stream:
        for line in stream:
            row=json.loads(line);frame=round(row['simulated_seconds']*50)
            if frame==0:
                continue
            if frame!=last+1 or not 1<=frame<len(positions):
                raise ValueError('diagnostic reference window missing, repeated or beyond endpoint')
            if abs(row['simulated_seconds']-frame/50)>1e-8:
                raise ValueError('diagnostic time is outside50Hz reference grid')
            qpos=np.asarray(row['qpos'],dtype=float)
            if qpos.shape!=(38,) or not np.isfinite(qpos).all():
                raise ValueError('invalid diagnostic pose')
            height=float(abs(qpos[2]-positions[frame,2]))
            gravity=float(np.sum((rotation(qpos[3:7])[2]-rotation(quaternions[frame])[2])**2))
            height_max=max(height_max,height);gravity_max=max(gravity_max,gravity)
            if first is None and (height>.17 or gravity>1.6):first=frame/50
            last=frame
    if last==0:raise ValueError('empty diagnostic trajectory')
    return dict(transitions_evaluated=last,simulated_seconds=last/50,
        first_failure_seconds=first,max_height_error_m=height_max,max_gravity_error_sq=gravity_max,
        full_reference_completed=last==len(positions)-1,
        standard_full_success=last==len(positions)-1 and first is None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=evaluate(args.truth,args.reference)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
