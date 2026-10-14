"""Measure surface pattern anchoring through world-position reprojection."""
import json
from pathlib import Path
import numpy as np
from inkforge.renderer import Renderer,Style
from inkforge.procedural import plane
from inkforge.scene import Scene,Material,Primitive
from inkforge.math3d import look_at,perspective


def sample(image,xy):
    x=np.clip(xy[:,0],0,image.shape[1]-1.001);y=np.clip(xy[:,1],0,image.shape[0]-1.001)
    ix=x.astype(int);iy=y.astype(int);fx=(x-ix)[:,None];fy=(y-iy)[:,None]
    return (image[iy,ix]*(1-fx)+image[iy,ix+1]*fx)*(1-fy)+(image[iy+1,ix]*(1-fx)+image[iy+1,ix+1]*fx)*fy


def main():
    v,i=plane(20);scene=Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'pattern reprojection plane')
    w,h=960,600;r=Renderer(scene,w,h,shadow_size=512)
    s=Style();eye=np.array([0.,3.,5.]);target=np.zeros(3);shift=np.array([.17,0,0])
    result={}
    for frequency in [4.,10.,32.]:
        s.dot_frequency=frequency
        r.render(eye,target,s,debug=7);first=r.output.to_numpy()[:,:,:2].copy();pos=r.gbuffers[0].to_numpy().copy()
        r.render(eye+shift,target+shift,s,debug=7);second=r.output.to_numpy()[:,:,:2].copy()
        extent=np.linalg.norm(scene.bounds[1]-scene.bounds[0])
        vp=perspective(55,w/h,max(.005,extent*.0001),max(extent*8,100))@look_at(eye+shift,target+shift)
        points=np.concatenate([pos[:,:,:3].reshape(-1,3),np.ones((w*h,1))],axis=1)
        clip=points@vp.T;ndc=clip[:,:2]/clip[:,3:4]
        xy=(ndc*np.array([.5,-.5])+.5)*[w,h]-.5
        y,x=np.mgrid[:h,:w];mask=(pos[:,:,3]>.5)&(y>h*.35)&(x>30)&(x<w-30)&(y<h-30)
        valid=mask.ravel()&(xy[:,0]>2)&(xy[:,0]<w-3)&(xy[:,1]>2)&(xy[:,1]<h-3)
        warped=sample(second,xy[valid]);reference=first.reshape(-1,2)[valid]
        stable=np.mean(abs(warped-reference),axis=0)
        unwarped=np.mean(abs(second.reshape(-1,2)[valid]-reference),axis=0)
        result[str(frequency)]={'reprojected_mean_absolute_error_dots_hatches':stable.tolist(),
            'unreprojected_error':unwarped.tolist(),'units':'0-1 display-encoded pattern mask',
            'samples':int(valid.sum()),'resolution':[w,h]}
    Path('output/pattern-stability.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
