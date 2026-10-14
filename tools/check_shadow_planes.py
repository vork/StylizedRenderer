"""Inspect receiver/caster tangent planes on a provably unoccluded height field."""
from pathlib import Path
import runpy
import numpy as np
from inkforge.renderer import Renderer,Style
from inkforge.environment import lighting_state
from inkforge.shadows import cascades
from inkforge.math3d import look_at


def unpack(value):
    bits=np.asarray(abs(value),dtype=np.uint32)-1
    o=np.stack([bits%4096,bits//4096],axis=-1)/4095*2-1
    n=np.concatenate([o,(1-abs(o).sum(axis=-1))[...,None]],axis=-1)
    folded=(1-abs(o[...,::-1]))*np.where(o>=0,1,-1)
    n[...,:2]=np.where((n[...,2]<0)[...,None],folded,o)
    return n/np.linalg.norm(n,axis=-1,keepdims=True)


def main():
    scene=runpy.run_path('tests/test_gpu.py')['bent_receiver_scene']()
    s=Style();r=Renderer(scene,320,200,shadow_size=512)
    eye=[0,2,4];target=[0,0,0]
    r.save('output/shadow-review/bent.png',eye,target,s,debug=9)
    p=r.gbuffers[0].to_numpy();shadows=[t.to_numpy() for t in r.shadows]
    mask=(p[:,:,3]!=0)&(abs(p[:,:,0])<1.6)&(abs(p[:,:,2])<1.6)
    srgb=r.output.to_numpy()[:,:,0][mask];linear=((srgb+.055)/1.055)**2.4
    print('LINEAR VISIBILITY mean / quantiles:',linear.mean(),np.percentile(linear,[0,1,5,50,100]))
    extent=np.linalg.norm(scene.bounds[1]-scene.bounds[0])
    matrices,splits=cascades(eye,target,55,1.6,max(.005,extent*.0001),20,
        lighting_state(scene,s)['direction'],scene.bounds,512)
    view=look_at(eye,target)
    for x,y in [(160,100),(140,90),(180,110)]:
        world=p[y,x,:3];n=unpack(p[y,x,3]);c=min(np.searchsorted(splits,-(view@np.r_[world,1])[2]),3)
        matrix=matrices[c];nm=np.linalg.inv(matrix[:3,:3]).T
        ndc=(matrix@np.r_[world,1])[:3];texel=(ndc[:2]*[.5,-.5]+.5)*512-.5
        pixel=np.floor(texel).astype(int);phase=texel-pixel;plane=nm@n;slope=-plane[:2]/plane[2]
        print('PIXEL',x,y,'cascade',c,'n',n,'GPU',r.output.to_numpy()[y,x,0])
        total=0.;lit=0.
        for yy in range(-2,4):
            for xx in range(-2,4):
                weight=np.clip(2.5-abs(xx-phase[0]),0,1)*np.clip(2.5-abs(yy-phase[1]),0,1)
                if weight<=0:continue
                tap=pixel+[xx,yy];sample=shadows[c][tap[1],tap[0]]
                cn=unpack(sample[1]);cp=nm@cn;cs=-cp[:2]/cp[2]
                offset=(tap+.5)/512*[2,-2]+[-1,1]-ndc[:2]
                receiver=ndc[2]+slope@offset
                agreement=abs(n@cn);a=np.clip((agreement-.94)/.05,0,1);a=a*a*(3-2*a)
                allowance=np.linalg.norm(slope-cs)*(np.linalg.norm(offset)+1/512)*a
                error=receiver-sample[0]
                total+=weight;lit+=weight*(error-allowance<.000015)
                if error-allowance>.000015:print('FAIL',xx,yy,'depth gap',error,'allowance',allowance,'agreement',agreement,'caster n',cn)
        print('CPU visibility',lit/total)


if __name__=='__main__':main()
