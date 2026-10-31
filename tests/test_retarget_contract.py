"""Focused tests for the numerical/coordinate contracts consumed by training."""
from pathlib import Path
import numpy as np
from retarget.processing import HUMAN_Y_UP_TO_ROBOT_Z_UP as C, strict_float32_angles
from retarget.yy.bvh_adapters import read_transforms
from retarget.yy.motion_postprocess import output_times


def test_geno_translation_channels_replace_offsets(tmp_path):
    p=tmp_path/'sample.bvh'
    p.write_text('''HIERARCHY
ROOT Hips
{
 OFFSET 100 200 300
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Foot
 {
  OFFSET 0 -100 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  End Site
  {
   OFFSET 0 0 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.016667
1 2 3 0 0 0 0 -10 0 0 0 0
1 2 3 0 0 0 0 -11 0 0 0 0
''')
    *_,pos,rot=read_transforms(p)
    np.testing.assert_allclose(pos[:,0],[[1,2,3],[1,2,3]])
    np.testing.assert_allclose(pos[:,1],[[1,-8,3],[1,-9,3]])


def test_coordinate_change_is_proper_rotation_and_maps_axes():
    np.testing.assert_allclose(C@C.T,np.eye(3));assert np.linalg.det(C)==1.
    np.testing.assert_allclose(C@np.array([0,1,0]),[0,0,1])
    np.testing.assert_allclose(C@np.array([1,0,0]),[0,1,0])


def test_hard_limit_float32_rounds_inside_original_bounds():
    limits=np.array([[-2.74,2.74],[-.175,2.443],[-.436,.436]])
    q=strict_float32_angles(np.array([limits[:,0],limits[:,1]]),limits)
    assert np.all(q.astype(float)>=limits[:,0]);assert np.all(q.astype(float)<=limits[:,1])


def test_complete_source_retiming_never_extrapolates():
    for n in [13541,6131,9835]:
        t=output_times(n,1/.016667,50.)
        assert t[-1] <= (n-1)*.016667
        assert (n-1)*.016667-t[-1] < .02
        np.testing.assert_allclose(np.diff(t),.02,atol=1e-13)
