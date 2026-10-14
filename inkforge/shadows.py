"""Four stable cascades with practical splits and overlapping receiver ranges."""
import numpy as np
from .math3d import look_at,normalize,orthographic


def cascades(eye,target,fov,aspect,near,distance,direction,bounds,resolution,count=4):
    ratios=np.arange(1,count+1)/count
    splits=.75*near*(distance/near)**ratios+.25*(near+(distance-near)*ratios)
    # Concentrate the near cascades together for large-scene fly-throughs.
    # Shrinking only the first split leaves an abrupt resolution cliff behind
    # it, reintroducing faceted false shadows on nearby folded receivers.
    # Small object scenes retain their broader practical split distribution.
    if splits[0]>2.:
        splits=.97*near*(distance/near)**ratios+.03*(near+(distance-near)*ratios)
    inv_view=np.linalg.inv(look_at(eye,target))
    low,high=bounds;extent=np.linalg.norm(high-low)
    # Fixed light orientation and depth range: off-screen casters are retained.
    lv=look_at(normalize(direction)*extent*2,np.zeros(3)).astype(np.float64)
    # Light eye is exactly on its Z axis. Avoid float32 dot-product cancellation
    # introducing a sub-texel XY translation after snapping a small near cascade.
    lv[:2,3]=0
    corners=np.array([[x,y,z,1] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    light_corners=corners@lv.T
    znear=-light_corners[:,2].max()-1;zfar=-light_corners[:,2].min()+1
    matrices=[]
    for index,end in enumerate(splits):
        previous=near if index==0 else splits[index-1]
        # Match the blend region sampled from the preceding cascade.
        start=near if index==0 else previous-(previous-(near if index==1 else splits[index-2]))*.12
        tan=np.tan(np.radians(fov)*.5)
        frustum=np.array([[x*z*tan*aspect,y*z*tan,-z,1] for z in (start,end) for x in (-1,1) for y in (-1,1)])
        world=frustum@inv_view.T;center=world[:,:3].mean(0)
        # Rotation-invariant bounding sphere avoids shimmering during rotation.
        radius=np.ceil(np.linalg.norm(world[:,:3]-center,axis=1).max()*16)/16
        radius*=resolution/(resolution-10) # four-texel PCF radius plus rounding guard
        lc=lv@np.append(center,1)
        texel=2*radius/resolution
        snapped=np.floor(lc[:2]/texel+.5)*texel
        matrices.append(orthographic(snapped[0]-radius,snapped[0]+radius,
            snapped[1]-radius,snapped[1]+radius,znear,zfar)@lv)
    return np.array(matrices,np.float32),np.asarray(splits,np.float32)
