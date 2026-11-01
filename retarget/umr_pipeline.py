"""Standalone native UMR NR-FBX pipeline for the new Sprite0825 embodiment."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import filecmp
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UMR=ROOT/'third_party/umr'
MOTIONS=('dance2_subject1',)
ROBOT_NAME='opensprite0825'
CORRESPONDENCE_DIR=ROOT/'assets/umr_correspondence/geno_opensprite0825'


def owned(path):
    path=Path(path).resolve()
    try:relative=path.relative_to(ROOT)
    except ValueError as exc:raise ValueError(f'Runtime path must be repository-owned: {path}') from exc
    if relative.parts and relative.parts[0]=='.reference':raise ValueError('.reference is inspection-only')
    return path


def stage_sources(motions=MOTIONS,cache_dir=None):
    """Copy unchanged selected BVHs and skinned FBX; no symlink or other checkout."""
    from retarget.bvh_tools.bvh_adapters import inspect_bvh,bvh_skeleton_signature,validate_geno_skeleton
    directory=owned(owned(cache_dir or ROOT/'outputs/umr/cache')/'source/geno')
    motion_directory=owned(directory/'motion_actor_retarget_205_with_ids');motion_directory.mkdir(parents=True,exist_ok=True)
    # Skeleton validation uses the repository-owned Geno stance.
    for name in motions:
        if name not in MOTIONS:raise ValueError(f'Unsupported source motion: {name}')
        source=owned(ROOT/'data/sources/lafan1-resolved'/f'{name}.bvh')
        info=inspect_bvh(source)
        if info['adapter']!='orangeduck_geno':raise ValueError(f'Not canonical Geno: {source}')
        validate_geno_skeleton(info,bvh_skeleton_signature(source).local_rest_offsets_m)
        target=owned(motion_directory/source.name)
        if target.exists() and not filecmp.cmp(source,target,shallow=False):
            raise ValueError(f'Cached BVH differs from the raw source; choose a fresh --cache-dir: {target}')
        if not target.exists():shutil.copyfile(source,target)
    fbx=owned(directory/'Geno.fbx')
    source_fbx=owned(ROOT/'assets/geno/Geno.fbx')
    if fbx.exists() and not filecmp.cmp(source_fbx,owned(fbx),shallow=False):
        raise ValueError(f'Cached FBX differs from the owned asset; choose a fresh --cache-dir: {fbx}')
    if not fbx.exists():shutil.copyfile(source_fbx,fbx)
    return directory


def check_node_runtime():
    """The parser uses the small checked-in Three.js runtime, never node_modules."""
    if shutil.which('node') is None:raise RuntimeError('Node.js >=18 is required for Geno FBX export')
    version=subprocess.check_output(['node','--version'],text=True).strip()
    if int(version.removeprefix('v').split('.')[0])<18:raise RuntimeError(f'Node.js >=18 required; found {version}')
    required=('nr_three_loader.mjs','export_nr_fbx_skin.mjs','FBXLoaderPatched.js',
              'vendor/three/package.json','vendor/three/LICENSE','vendor/three/build/three.module.js',
              'vendor/three/examples/jsm/libs/fflate.module.js',
              'vendor/three/examples/jsm/curves/NURBSCurve.js',
              'vendor/three/examples/jsm/curves/NURBSUtils.js')
    for relative in required:
        path=owned(UMR/'scripts'/relative)
        if not path.is_file():raise FileNotFoundError(f'Missing vendored Node parser file: {path}')


def native_modules():
    for directory in (UMR,UMR/'scripts'):
        if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    import humanoid_retarget_pipeline as pipeline
    return pipeline


def config_for(name,run_dir,max_frames=0,start_frame=0,*,cache_dir=None,
               correspondence_dir=None,correspondence_device='cuda'):
    pipeline=native_modules()
    config=pipeline.load_pipeline_config(ROOT/'configs/umr/opensprite.json',UMR/'humanoid_retarget_defaults_nr.json')
    robot=config['robot']
    if robot['name']!=ROBOT_NAME or robot.get('slot_name',ROBOT_NAME)!=ROBOT_NAME:
        raise ValueError('Only the new OpenSprite correspondence identity is accepted')
    xml=owned(pipeline.resolve_path(robot['xml'],config))
    expected=ROOT/'assets/opensprite/retarget_scene.xml'
    if xml!=expected:raise ValueError(f'Unexpected robot model: {xml}')
    robot['xml']=str(xml)
    # The official repository-owned MJCF already has the real floating root.
    # Prevent an upstream convenience path from writing a generated model beside it.
    robot.setdefault('xml_policy',{})['add_freejoint_root']=False
    if robot.get('point_cloud_center')!='body:pelvis_link':raise ValueError('OpenSprite center contract changed')
    if config['correspondence']['dataset'].get('smpl_center')!='pelvis':raise ValueError('Geno Hips center contract changed')
    config['smpl_template']={'name':'geno','source':'motion','type':'nr_fbx','use_betas':False,'use_gender':False,'betas':None}
    config['motion']={'data':str(stage_sources((name,),cache_dir)),'seq_key':name,'start':start_frame,'end':-1,'stride':1,'max_frames':max_frames}
    # Native dq is integrated with dt=1. Its existing per-dof box is applied
    # at each solver iteration; divide the source-frame velocity allowance
    # by iteration count to bound the complete physical frame transition.
    # global_step_size is an L2 cone radius, not a multiplicative update factor.
    from retarget.processing import robot_contract
    from retarget.bvh_tools.bvh_adapters import inspect_bvh
    _,joint_names,_,velocity=robot_contract()
    frame_time=inspect_bvh(ROOT/'data/sources/lafan1-resolved'/f'{name}.bvh')['frame_time']
    iterations=int(config['solver']['iters'])
    robot['dof_max_dq_box']={joint:float(v*frame_time*.95/iterations) for joint,v in zip(joint_names,velocity)}
    corr=config['correspondence'];directory=owned(correspondence_dir or CORRESPONDENCE_DIR)
    corr['smpl_name']='geno'
    corr['slots']=str(directory/'training/correspondence_slots_final.npz')
    corr.setdefault('dataset',{})['out']=str(directory/'dataset.npz')
    corr.setdefault('train',{}).update(template_name='geno',out_dir=str(directory/'training'),device=correspondence_device)
    config.setdefault('retarget',{}).update(out=str(owned(run_dir)/name/'native.npz'))
    config['view']={'enabled':False,'play':False}
    destination=owned(run_dir)/name/'config.json';destination.parent.mkdir(parents=True,exist_ok=True)
    # Native config path ownership remains with this explicit generated file.
    config.pop('_config_path',None);config.pop('_config_dir',None)
    destination.write_text(json.dumps(config,indent=2)+'\n')
    return destination


def validate_slots(correspondence_dir=None):
    path=owned(correspondence_dir or CORRESPONDENCE_DIR)/'training/correspondence_slots_final.npz'
    path=owned(path)
    with np.load(path,allow_pickle=True) as archive:
        names=np.asarray(archive['names']).astype(str).tolist()
        if names!=['geno',ROBOT_NAME]:raise ValueError(f'Wrong robot/source correspondence: {names}')
        if str(np.asarray(archive['training_mode']).item())!='all_samples':raise ValueError('Unexpected native NR correspondence training mode')
        slots=np.asarray(archive['reconstructed_slots'])
        if slots.shape!=(2,4096,3) or not np.isfinite(slots).all():raise ValueError('Invalid new OpenSprite surface slots')
    return path


def run_stage(stage,config_path,*,record_path=None):
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    environment['PYTHONPATH']=str(UMR)+os.pathsep+str(UMR/'scripts')
    if stage!='train':environment['CUDA_VISIBLE_DEVICES']=''
    command=[sys.executable,str(UMR/'scripts/humanoid_retarget_pipeline.py'),'--config',str(config_path),'--defaults',str(UMR/'humanoid_retarget_defaults_nr.json'),'--stage',stage,'--skip-view']
    if stage=='retarget':command.append('--force-retarget')
    record_path=owned(record_path or ROOT/'outputs/umr/correspondence_gpu_time.json')
    record=None;attempt=None
    start=time.monotonic()
    if stage=='train':
        record_path.parent.mkdir(parents=True,exist_ok=True)
        record=json.loads(record_path.read_text()) if record_path.exists() else {'stage':'native_umr_correspondence','attempts':[]}
        if 'attempts' not in record:
            # Preserve the first attempt produced before cumulative accounting.
            record={'stage':'native_umr_correspondence','attempts':[record]}
        now=datetime.now(timezone.utc)
        for prior in record['attempts']:
            if prior.get('status')=='running':
                since=datetime.fromisoformat(prior['started_at'])
                prior['gpu_training_seconds']=max(float(prior.get('gpu_training_seconds',0)),(now-since).total_seconds())
                prior['status']='interrupted_conservative_wallclock_charge'
        attempt={'started_at':now.isoformat(),'status':'running','gpu_training_seconds':0.}
        record['attempts'].append(attempt)
        record['gpu_training_seconds']=sum(float(a.get('gpu_training_seconds',0)) for a in record['attempts'])
        temporary=record_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(record_path)
    result=None;child=None
    def stop_handler(signum,frame):
        raise KeyboardInterrupt(f'Native UMR interrupted by signal {signum}')
    previous_term=signal.signal(signal.SIGTERM,stop_handler)
    try:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,start_new_session=True)
        returncode=child.wait()
        result=child
        if returncode:raise RuntimeError(f'Native UMR stage {stage} failed with code {returncode}')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        signal.signal(signal.SIGTERM,previous_term)
        elapsed=time.monotonic()-start
        if attempt is not None:
            attempt.update(finished_at=datetime.now(timezone.utc).isoformat(),status='complete' if result is not None and result.returncode==0 else 'failed_or_interrupted',gpu_training_seconds=elapsed)
            record['gpu_training_seconds']=sum(float(a.get('gpu_training_seconds',0)) for a in record['attempts'])
            temporary=record_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(record_path)
            print(f"[BUDGET] native UMR attempt charged {elapsed:.3f}s; cumulative {record['gpu_training_seconds']:.3f}s",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['build','train','retarget','convert','all'],default='all')
    parser.add_argument('--motion','--motions',dest='motions',nargs='+',choices=['dance2_subject1'],default=['dance2_subject1'],help='Complete released dance. --motions is a compatibility alias.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/umr/rebuild',help='Native solver output; repository-owned, separate from published data')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/umr/cache',help='Unchanged staged Geno FBX/BVH input cache')
    parser.add_argument('--motion-output-dir',type=Path,help='Converted individual trajectories and packed bank; default OUTPUT_DIR/motions')
    parser.add_argument('--correspondence-dir',type=Path,default=CORRESPONDENCE_DIR,help='Bundled correspondence by default; choose a fresh outputs/ directory to rebuild it')
    parser.add_argument('--correspondence-device',choices=['cpu','cuda'],default='cuda',help='Only correspondence training uses this device; native retargeting is CPU-only')
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--max-frames',type=int,default=0)
    args=parser.parse_args(argv)
    if args.start_frame<0 or args.max_frames<0:parser.error('Frame arguments must be nonnegative')
    if len(set(args.motions))!=len(args.motions):parser.error('Select each motion at most once')
    if args.stage=='convert' and (args.start_frame or args.max_frames):parser.error('Production conversion requires the complete source; use retarget for a partial CPU smoke')
    args.output_dir=owned(args.output_dir);args.cache_dir=owned(args.cache_dir)
    args.correspondence_dir=owned(args.correspondence_dir)
    motion_output=owned(args.motion_output_dir or args.output_dir/'motions')
    options=dict(cache_dir=args.cache_dir,correspondence_dir=args.correspondence_dir,correspondence_device=args.correspondence_device)
    ledger=args.output_dir/'correspondence_gpu_time.json'
    if args.stage!='convert':
        check_node_runtime()
        first=config_for(args.motions[0],args.output_dir,args.max_frames,args.start_frame,**options)
    if args.stage in ('build','all'):run_stage('build',first,record_path=ledger)
    if args.stage in ('train','all'):run_stage('train',first,record_path=ledger)
    if args.stage in ('retarget','all'):
        slots=validate_slots(args.correspondence_dir)
        for name in args.motions:
            path=config_for(name,args.output_dir,args.max_frames,args.start_frame,**options)
            # Retarget-only must never silently fall through to GPU training.
            pipeline=native_modules()
            config=pipeline.load_pipeline_config(path,UMR/'humanoid_retarget_defaults_nr.json')
            if not pipeline.correspondence_slots_compatible(slots,config):
                raise ValueError('Correspondence is incompatible; explicitly run build/train in a fresh correspondence directory')
            run_stage('retarget',path,record_path=ledger)
    if args.stage=='convert' or (args.stage=='all' and args.start_frame==0 and args.max_frames==0):
        from retarget.convert_umr import convert,pack
        paths=[convert(owned(args.output_dir/name/'native.npz'),name,owned(motion_output/f'{name}.npz'),correspondence_dir=args.correspondence_dir) for name in args.motions]
        pack(paths,owned(motion_output/'lafan_dance.npz'),correspondence_dir=args.correspondence_dir)

if __name__=='__main__':main()
