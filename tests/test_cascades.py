import numpy as np
from inkforge.shadows import cascades
from inkforge.math3d import look_at


def test_cascade_frusta_are_covered_and_texel_snapped():
    eye=np.array([7.,2.3,0]);target=np.array([-5.,4,0]);near=.01;distance=80;fov=55;aspect=1.6
    bounds=(np.array([-40,-5,-60]),np.array([70,30,60]))
    matrices,splits=cascades(eye,target,fov,aspect,near,distance,[-.5,.8,.35],bounds,2048)
    assert np.all(np.diff(splits)>0) and abs(splits[-1]-distance)<1e-5
    inverse=np.linalg.inv(look_at(eye,target));start=near
    for matrix,end in zip(matrices,splits):
        t=np.tan(np.radians(fov)/2)
        corners=np.array([[x*z*t*aspect,y*z*t,-z,1] for z in [start,end] for x in [-1,1] for y in [-1,1]])@inverse.T
        clip=corners@matrix.T
        assert np.abs(clip[:,:2]).max()<1.
        texels=matrix[:2,3]*1024
        np.testing.assert_allclose(texels,np.round(texels),atol=.002)
        start=end
