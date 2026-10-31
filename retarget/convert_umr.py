"""Convert complete native UMR trajectories to one guarded OpenSprite motion bank."""
import argparse
import json
from pathlib import Path
import numpy as np
from retarget.processing import robot_contract,refine_angles,kinematics,geometry_quality,correct_small_ground_residual
from retarget.yy.motion_postprocess import process_motion
from retarget.yy.bvh_adapters import inspect_bvh
from retarget.umr_pipeline import ROOT,MOTIONS,validate_slots

METHOD='umr_native_nr_surface_vector_v1'
REVISION='opensprite-umr-geno-v1-20261031'


def guard_metadata():
    path=validate_slots()
    with np.load(path,allow_pickle=True) as slots:names=np.asarray(slots['names']).astype(str)
    return dict(retarget_method=np.asarray(METHOD),robot_model=np.asarray('opensprite0825'),dataset_revision=np.asarray(REVISION),correspondence_names=names)


def convert(source,name,output,allow_partial=False):
    model,joints,limits,velocity=robot_contract()
    source_info=inspect_bvh(ROOT/'data/sources/lafan1-resolved'/f'{name}.bvh')
    with np.load(source,allow_pickle=True) as data:
        if str(np.asarray(data['robot_name']).item())!='opensprite0825':raise ValueError('Native UMR target is not OpenSprite')
        if str(np.asarray(data['source_format']).item())!='nr_fbx_bvh':raise ValueError('Native UMR source is not Geno NR surface')
        if np.asarray(data['robot_joint_names']).astype(str).tolist()!=joints:raise ValueError('Native UMR joint order differs from official Sprite model')
        qpos=np.asarray(data['qpos'],dtype=np.float64)
        frame_ids=np.asarray(data['frame_ids'])
    expected=np.arange(source_info['frames'])
    if not allow_partial and (len(qpos)!=len(expected) or not np.array_equal(frame_ids,expected)):
        raise ValueError('Production conversion requires every native frame in source order')
    if qpos.shape!=(len(frame_ids),model.nq) or not np.isfinite(qpos).all():raise ValueError('Invalid native qpos')
    if not np.allclose(np.linalg.norm(qpos[:,3:7],axis=1),1.,rtol=0,atol=1e-6):raise ValueError('Invalid native root quaternions')
    excess=np.maximum(limits[:,0]-qpos[:,7:],qpos[:,7:]-limits[:,1])
    if excess.max()>1e-6:raise ValueError(f'Native UMR hard joint residual exceeds roundoff: {excess.max()} rad')
    residual_count=int(np.count_nonzero(excess>0))
    qpos[:,7:]=np.clip(qpos[:,7:],limits[:,0],limits[:,1])
    raw_quality=geometry_quality(model,qpos,source_info['fps'])
    qpos,times,resampling=process_motion(qpos,source_info['fps'],50.,model=model)
    qpos[:,7:],refinement=refine_angles(qpos[:,7:],limits,velocity)
    qpos,ground_correction=correct_small_ground_residual(model,qpos)
    quality=geometry_quality(model,qpos,50.)
    if quality['minimum_collision_z_m']<0:raise ValueError('Final float32 ground geometry failed')
    payload=kinematics(model,qpos)
    metadata=guard_metadata()
    payload.update(metadata,qpos=qpos,time=times,fps=np.asarray(50.),joint_names=np.array(joints),body_names=np.array([model.body(i).name for i in range(1,model.nbody)]),source_name=np.asarray(name),source_fps=np.asarray(source_info['fps']),source_frames=np.asarray(len(frame_ids)),coordinate_system=np.asarray('mujoco_z_up_wxyz'),motion_quality=np.asarray('needs_review'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**payload)
    qa=dict(motion=name,retarget_method=METHOD,robot_model='opensprite0825',dataset_revision=REVISION,source_frames=len(frame_ids),output_frames=len(qpos),fps=50.,duration_seconds=len(qpos)/50.,native_roundoff_residual_count=residual_count,native_geometry=raw_quality,final_geometry=quality,ground_correction=ground_correction,resampling=resampling,refinement=refinement,semantic_self_contact_map='disabled_by_native_NR_uniform_source_path')
    output.with_suffix('.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa),flush=True)
    return output


def merge(paths,output):
    clips=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:clips.append({k:data[k] for k in data.files})
    if [str(c['source_name']) for c in clips]!=list(MOTIONS):raise ValueError('Production bank requires exactly the three selected sources in canonical order')
    for clip in clips:
        if str(clip['retarget_method'])!=METHOD:raise ValueError('Prototype or other retarget method cannot enter production bank')
        for key in ('joint_names','body_names','correspondence_names'):
            if not np.array_equal(clip[key],clips[0][key]):raise ValueError(f'Incompatible {key}')
    fields=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')
    result={k:np.concatenate([c[k] for c in clips],axis=0) for k in fields}
    lengths=np.array([len(c['joint_pos']) for c in clips],np.int64)
    result.update(guard_metadata(),fps=np.asarray(50.),motion_names=np.array(MOTIONS),motion_starts=np.r_[0,np.cumsum(lengths)[:-1]],motion_lengths=lengths,joint_names=clips[0]['joint_names'],body_names=clips[0]['body_names'],motion_categories=np.array(['lafan/dance','lafan/getup','lafan/walk']),motion_qualities=np.array([str(c['motion_quality']) for c in clips]))
    np.savez_compressed(output,**result)
    print(json.dumps({'output':str(output),'names':list(MOTIONS),'lengths':lengths.tolist(),'starts':result['motion_starts'].tolist(),'retarget_method':METHOD}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/umr/full')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/motions')
    args=parser.parse_args()
    paths=[convert(args.input_dir/name/'native.npz',name,args.output_dir/f'{name}.npz') for name in MOTIONS]
    merge(paths,args.output_dir/'lafan_three.npz')

if __name__=='__main__':main()
