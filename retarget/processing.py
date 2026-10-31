"""Robot-owned motion postprocessing; no retargeting or other-robot model logic."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
HUMAN_Y_UP_TO_ROBOT_Z_UP=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])

def strict_float32_angles(angles, limits):
    """Keep rounding at official non-float32 hard bounds toward the interior."""
    lo = limits[:, 0].astype(np.float32)
    hi = limits[:, 1].astype(np.float32)
    lo = np.where(lo.astype(float) < limits[:, 0], np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(float) > limits[:, 1], np.nextafter(hi, np.float32(-np.inf)), hi)
    return np.clip(np.asarray(angles, np.float32), lo, hi)

def refine_angles(angles, limits, velocity_limits):
    """YY TrajOpt temporal regularizer with exact per-joint box/speed QPs.

    The temporal_penalty coefficients are copied from YY motion_refinement.py.
    Root samples/time are untouched. Positive safety margin absorbs solver and
    float32 roundoff; actual final closed-bound checks remain strict.
    """
    import math
    import osqp
    from scipy import sparse
    count=len(angles);fps=50.;weight=1e-7
    coefficients=[(-1)**(2-i)*math.comb(2,i) for i in range(3)]
    difference=sparse.diags([np.full(count-2,value,dtype=float) for value in coefficients],range(3),shape=(count-2,count),format='csc')
    derivative=math.sqrt(weight)*fps**2*difference
    hessian=sparse.eye(count,format='csc') + derivative.T@derivative
    d1=sparse.diags([-np.ones(count-1),np.ones(count-1)],[0,1],shape=(count-1,count),format='csc')
    constraint=sparse.vstack([sparse.eye(count,format='csc'),d1],format='csc')
    result=np.empty_like(angles);statuses=[]
    for joint in range(angles.shape[1]):
        speed=.98*velocity_limits[joint]/fps
        lower=np.r_[np.full(count,limits[joint,0]+1e-5),np.full(count-1,-speed)]
        upper=np.r_[np.full(count,limits[joint,1]-1e-5),np.full(count-1,speed)]
        qp=osqp.OSQP();qp.setup(P=sparse.triu(hessian).tocsc(),q=-angles[:,joint],A=constraint,l=lower,u=upper,verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,polishing=False)
        solution=qp.solve();statuses.append(solution.info.status)
        if solution.info.status_val not in (1,2):raise RuntimeError(f'Joint {joint} refinement failed: {solution.info.status}')
        result[:,joint]=solution.x
    result=strict_float32_angles(result,limits)
    speed=np.abs(np.diff(result.astype(float),axis=0))*fps
    if np.any(speed>velocity_limits+1e-6):raise ValueError('Final float32 velocity violation')
    if np.any(result.astype(float)<limits[:,0]) or np.any(result.astype(float)>limits[:,1]):raise ValueError('Final float32 hard joint violation')
    return result,dict(method='YY physical-time acceleration QP, per-joint box and speed constraints',acceleration_weight=weight,velocity_limit_margin=.98,max_correction_rad=float(np.abs(result-angles).max()),rms_correction_rad=float(np.sqrt(np.mean((result-angles)**2))),max_velocity_ratio=float(np.max(speed/velocity_limits)),constraints_passed=True)

def kinematics(model,qpos,fps=50.):
    """World FK and velocities within exactly one source trajectory."""
    data=mujoco.MjData(model);n=len(qpos);b=model.nbody-1
    if n<3:raise ValueError('At least three samples required for velocity derivation')
    p=np.empty((n,b,3));q=np.empty((n,b,4));qvel=np.empty((n,model.nv))
    for i in range(n):
        data.qpos[:]=qpos[i];mujoco.mj_kinematics(model,data)
        p[i]=data.xpos[1:];q[i]=data.xquat[1:]
        a=max(0,i-1);z=min(n-1,i+1)
        mujoco.mj_differentiatePos(model,qvel[i],(z-a)/fps,qpos[a],qpos[z])
    lin=np.gradient(p,1./fps,axis=0,edge_order=2)
    ang=np.zeros((n,b,3))
    for j in range(b):
        r=Rotation.from_quat(q[:,j][:,[1,2,3,0]])
        ang[1:-1,j]=(r[2:]*r[:-2].inv()).as_rotvec()/(2./fps)
        ang[0,j]=(r[1]*r[0].inv()).as_rotvec()*fps
        ang[-1,j]=(r[-1]*r[-2].inv()).as_rotvec()*fps
    return {k:v.astype(np.float32) for k,v in dict(joint_pos=qpos[:,7:],joint_vel=qvel[:,6:],qvel=qvel,body_pos_w=p,body_quat_w=q,body_lin_vel_w=lin,body_ang_vel_w=ang).items()}


def robot_contract():
    """Read every joint from the Sprite0825 model and official URDF."""
    model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/opensprite/opensprite.xml'))
    names=[model.joint(i).name for i in range(1,model.njnt)]
    urdf=ROOT/'assets/opensprite/vendor/sprite0825/sprite0825_sanitized_v4.urdf'
    official={j.get('name'):j.find('limit') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute'}
    if set(names)!=set(official):raise ValueError('OpenSprite MJCF/URDF joint identities differ')
    limits=np.array([[float(official[n].get('lower')),float(official[n].get('upper'))] for n in names])
    if not np.allclose(model.jnt_range[1:],limits,rtol=0,atol=1e-12):raise ValueError('OpenSprite MJCF/URDF hard joint limits differ')
    velocity=np.array([float(official[n].get('velocity')) for n in names])
    return model,names,limits,velocity


def geometry_quality(model,qpos,fps=50.):
    """Measure actual primitive support and allowed-pair self contacts."""
    data=mujoco.MjData(model)
    ids=np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0))
    minima=np.empty(len(qpos));self_min=np.zeros(len(qpos));pairs={}
    for i,pose in enumerate(qpos):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        heights=data.geom_xpos[ids,2].copy()
        for j,gid in enumerate(ids):
            z=data.geom_xmat[gid].reshape(3,3)[2];size=model.geom_size[gid];kind=model.geom_type[gid]
            if kind==mujoco.mjtGeom.mjGEOM_BOX:heights[j]-=np.abs(z)@size
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:heights[j]-=size[0]+abs(z[2])*size[1]
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:heights[j]-=size[0]
            else:raise ValueError(f'Unsupported OpenSprite collision geometry {kind}')
        minima[i]=heights.min()
        mujoco.mj_collision(model,data)
        for contact in data.contact[:data.ncon]:
            if contact.dist>=0:continue
            self_min[i]=min(self_min[i],float(contact.dist))
            pair=' / '.join([model.geom(int(contact.geom1)).name,model.geom(int(contact.geom2)).name])
            pairs[pair]=min(pairs.get(pair,0.),float(contact.dist))
    mask=minima>.02
    edge=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    return dict(minimum_collision_z_m=float(minima.min()),minimum_ground_time_s=float(minima.argmin()/fps),ground_fraction_below_minus_1cm=float(np.mean(minima<-.01)),self_minimum_signed_distance_m=float(self_min.min()),self_worst_time_s=float(self_min.argmin()/fps),self_fraction_below_minus_1cm=float(np.mean(self_min<-.01)),self_fraction_below_minus_3cm=float(np.mean(self_min<-.03)),all_body_gap_over_2cm_seconds=float(mask.sum()/fps),all_body_gap_longest_seconds=float(np.max(ends-starts)/fps) if len(starts) else 0.,worst_self_pairs=dict(sorted(pairs.items(),key=lambda x:x[1])[:10]))


def correct_small_ground_residual(model,qpos,cap=.03,clearance=.0005):
    """Reconcile small measured post-filter primitive residuals, capped at 3 cm.

    This is an affine world-Z translation only; it cannot conceal a bad native
    solve or repair self collisions. Exceeding the cap rejects the trajectory.
    """
    import osqp
    from scipy import sparse
    data=mujoco.MjData(model);n=len(qpos)
    ids=np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0))
    minimum=np.empty(n)
    for i,pose in enumerate(qpos):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        heights=data.geom_xpos[ids,2].copy()
        for j,gid in enumerate(ids):
            z=data.geom_xmat[gid].reshape(3,3)[2];size=model.geom_size[gid];kind=model.geom_type[gid]
            if kind==mujoco.mjtGeom.mjGEOM_BOX:heights[j]-=np.abs(z)@size
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:heights[j]-=size[0]+abs(z[2])*size[1]
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:heights[j]-=size[0]
            else:raise ValueError(f'Unsupported collision primitive {kind}')
        minimum[i]=heights.min()
    lower=np.maximum(0.,clearance-minimum)
    if lower.max()>cap:
        frame=int(lower.argmax())
        raise ValueError(f'Native/postprocessed geometry needs {lower[frame]:.6f}m lift at frame {frame} ({frame/50.:.2f}s), exceeding {cap:.3f}m cap; review native solve instead')
    diff=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')*50.**2
    hessian=sparse.eye(n,format='csc')+1e-5*(diff.T@diff)
    qp=osqp.OSQP();qp.setup(P=sparse.triu(hessian).tocsc(),q=np.zeros(n),A=sparse.eye(n,format='csc'),l=lower,u=np.full(n,cap),verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=30000,polishing=False)
    solved=qp.solve()
    if solved.info.status_val not in (1,2):raise RuntimeError(f'Small ground residual QP failed: {solved.info.status}')
    lift=np.minimum(cap,np.maximum(solved.x,lower+1e-7))
    result=qpos.copy();result[:,2]+=lift
    result=result.astype(np.float32)
    return result,dict(method='small capped primitive world-Z residual QP after native UMR and joint-speed refinement',maximum_correction_m=float(lift.max()),mean_correction_m=float(lift.mean()),p95_correction_m=float(np.percentile(lift,95)),maximum_allowed_m=cap,minimum_collision_z_before_m=float(minimum.min()),clearance_m=clearance)
