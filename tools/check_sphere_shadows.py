"""Diagnose shadow taps that darken the lit side of an isolated smooth sphere."""
import numpy as np
import argparse
import json
from pathlib import Path
from inkforge.renderer import Renderer,Style
from inkforge.scene import Scene,Primitive,Material
from inkforge.procedural import sphere,showroom
from inkforge.environment import lighting_state
from inkforge.math3d import look_at
from inkforge.shadows import cascades
from tools.check_shadow_planes import unpack


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--showroom',action='store_true')
    parser.add_argument('--capture',type=Path);parser.add_argument('--hour',type=float)
    parser.add_argument('--pixel',nargs=2,type=int,action='append');parser.add_argument('--rays',action='store_true');args=parser.parse_args()
    v,i=sphere(np.zeros(3),radius=1)
    scene=Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Sphere shadow probe')
    if args.showroom:scene=showroom()
    width,height=(1280,800) if args.showroom else (640,400)
    s=Style(penumbra_strength=0)
    eye=[2,1,-4];target=[0,0,0]
    if args.showroom:eye=[4,4,-7];target=[0,.5,1]
    if args.capture:
        from inkforge.scene import load_scene
        data=json.loads(args.capture.read_text());scene=load_scene(data['source'])
        width,height=data['render_resolution'];eye=data['eye'];target=data['target'];s=Style(**data['style']);s.penumbra_strength=0
    if args.hour is not None:s.time_of_day=args.hour
    r=Renderer(scene,width,height,shadow_size=2048)
    r.render(eye,target,s,debug=11)
    p=r.gbuffers[0].to_numpy();output=r.output.to_numpy()[:,:,0]
    o=r.gbuffers[4].to_numpy()[:,:,:2]
    n=np.concatenate([o,(1-abs(o).sum(axis=2))[:,:,None]],axis=2)
    n[:,:,:2]=np.where((n[:,:,2]<0)[:,:,None],(1-abs(o[:,:,::-1]))*np.where(o>=0,1,-1),o)
    n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-8)
    l=lighting_state(scene,s)['direction'];nl=n@l
    mask=(p[:,:,3]!=0)&(nl>.16)&(output<.99)
    if args.showroom:mask&=p[:,:,1]>.2
    print('lit-side errors',mask.sum(),'minimum',output[mask].min() if mask.any() else 1)
    yx=np.argwhere(mask);yx=yx[np.argsort(output[mask])[:3]]
    if args.pixel:yx=np.array(args.pixel)[:,::-1]
    extent=np.linalg.norm(scene.bounds[1]-scene.bounds[0]);near=max(.005,extent*.0001)
    matrices,splits=cascades(eye,target,55,width/height,near,min(100,max(extent,20)),l,scene.bounds,2048)
    view=look_at(eye,target);maps=[t.to_numpy() for t in r.shadows]
    for y,x in yx:
        world=p[y,x,:3];n=unpack(p[y,x,3]);c=min(np.searchsorted(splits,-(view@np.r_[world,1])[2]),3)
        m=matrices[c];im=np.linalg.inv(m[:3,:3]);nm=im.T;ndc=(m@np.r_[world,1])[:3]
        texel=(ndc[:2]*[.5,-.5]+.5)*2048-.5;pixel=np.floor(texel).astype(int);phase=texel-pixel
        plane=nm@n;slope=-plane[:2]/plane[2] if abs(plane[2])>np.linalg.norm(plane)*1e-6 else np.zeros(2)
        rn=n if n@l>=0 else -n
        print('PIXEL',x,y,'visibility',output[y,x],'nl',nl[y,x],'cascade',c,'world',world,'face cos',rn@l,'plane',plane)
        if args.rays:
            nearest=float('inf');hit_info=None
            for primitive in scene.primitives:
                for start in range(0,len(primitive.indices),65535):
                    tri=primitive.vertices[primitive.indices[start:start+65535].reshape(-1,3),:3].astype(np.float64)
                    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
                    h=np.cross(l,e2);det=np.einsum('ij,ij->i',e1,h)
                    inv=1/np.where(abs(det)>1e-12,det,1e-12)
                    delta=world-tri[:,0];u=np.einsum('ij,ij->i',delta,h)*inv
                    q=np.cross(delta,e1);v=(q@l)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
                    valid=(abs(det)>1e-12)&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-4)&(t<nearest)
                    if valid.any():
                        ids=np.flatnonzero(valid);index=ids[np.argmin(t[valid])];nearest=t[index]
                        hit_info=(scene.materials[primitive.material].name,(start//3)+int(index))
            print('EXACT RAY closest blocker',nearest,hit_info,flush=True)
        for yy in range(0,2):
            for xx in range(0,2):
                tap=pixel+[xx,yy];depth,packed=maps[c][tap[1],tap[0]]
                cn=unpack(packed);agreement=abs(n@cn);cn=cn if cn@l>=0 else -cn
                offset=(tap+.5)/2048*[2,-2]+[-1,1]-ndc[:2];delta=im@np.r_[offset,depth-ndc[2]]
                radius=np.linalg.norm(im@np.r_[offset,0])+np.linalg.norm(im@np.array([1/2048,1/2048,0]))
                support=radius/max(min(abs(n@l),abs(cn@l)),.05)
                receiver=ndc[2]+slope@offset
                print('TAP',xx,yy,'gap',receiver-depth,'agreement',agreement,'halfspaces',rn@delta,cn@-delta,
                    'distance/support',np.linalg.norm(delta),support*1.5,'caster cos',cn@l)


if __name__=='__main__':main()
