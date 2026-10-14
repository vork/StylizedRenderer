"""Static/panning subpixel rails against a supersampled linear-light reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from inkforge.scene import Scene,Material,Primitive,make_vertices
from inkforge.renderer import Renderer,Style


def rails_scene(backdrop=True):
    def quad(x0,x1,y0,y1,z,slant=0):
        return make_vertices([[x0,y0,z],[x1,y0,z],[x1+slant,y1,z],[x0+slant,y1,z]],[0,1,2,0,2,3])
    v,i=quad(-40,40,-20,20,-5)
    primitives=[Primitive(v,i,0)] if backdrop else []
    for index,x in enumerate(np.linspace(-9,9,25)):
        width=(.025,.05,.1)[index%3]
        v,i=quad(x,x+width,-4,4,0,.25 if index%2 else 0)
        primitives.append(Primitive(v,i,1))
    return Scene(primitives,[Material(base=[.65,.65,.65,1],unlit=True,double_sided=True),
                            Material(base=[.01,.01,.01,1],unlit=True,double_sided=True)],[],'Distant rails')


def measure(backdrop=True,motion_jitter=.25):
    scene=rails_scene(backdrop);w,h=480,300
    r=Renderer(scene,w,h,shadow_size=128)
    r.motion_jitter_scale=motion_jitter
    reference=Renderer(scene,w*4,h*4,shadow_size=128,device=r.device)
    s=Style(bloom=0,fog=0,outline_wobble=0,hatching=0,halftone=0,exposure=0)
    ref=Style(**vars(s));ref.taa=False;ref.outline*=4
    # Restrict to the interior of the rails, away from their endpoints.
    mask=np.zeros((h,w),bool);mask[125:175,100:380]=True
    report={};images={}
    for mode,speed in [('stationary',0.),('pan',.025),('fast_pan',.24)]:
        r.taa_key=None;frames=[];errors=[];worst=[];ghosts=[]
        for frame in range(64):
            offset=(frame-32)*speed
            eye=[offset,0,32];target=[offset,0,0]
            r.render(eye,target,s)
            if frame>=32:
                actual=r.antialiased.to_numpy()[:,:,:3].astype(np.float32)
                frames.append(actual)
                if frame%4==0:
                    reference.render(eye,target,ref)
                    truth=reference.mapped.to_numpy()[:,:,:3].astype(np.float32).reshape(h,4,w,4,3).mean((1,3))
                    errors.append(float(abs(actual-truth)[mask].mean()))
                    worst.append(float(np.percentile(abs(actual-truth)[mask],95)))
                    # Even 4x sampling can miss a 0.225-pixel rail. Use its
                    # analytic projected support for the no-trails check,
                    # including its one-pixel ink border and two-pixel guard.
                    yy,xx=np.mgrid[:h,:w];f=h/(2*32*np.tan(np.radians(55)*.5))
                    world_y=(h*.5-yy-.5)/f
                    support=np.zeros((h,w),bool)
                    for index,x in enumerate(np.linspace(-9,9,25)):
                        width=(.025,.05,.1)[index%3];slant=.25 if index%2 else 0
                        center=(x+width*.5+slant*(world_y+4)/8-offset)*f+w*.5
                        support|=(abs(xx+.5-center)<width*f*.5+3)&(abs(world_y)<4+3/f)
                    clear=mask&~support
                    ghosts.append(float(np.percentile(abs(actual-truth)[clear],99)) if clear.any() else 0.)
                    images[mode]=actual
                    images['reference']=truth
        frames=np.asarray(frames)
        temporal=frames.std(axis=0)[mask]
        report[mode]={
            'temporal_std_mean':float(temporal.mean()),'temporal_std_p95':float(np.percentile(temporal,95)),
            'reference_mae':float(np.mean(errors)),'reference_error_p95':float(np.mean(worst)),
            'clear_background_error_p99':float(np.max(ghosts))}
        if mode=='stationary':images['variation']=frames.std(axis=0)*5
    # A full camera cut must clear all accumulated thin-line coverage.
    r.render([80,0,32],[80,0,0],s)
    report['cut_history_samples']=r.taa_frame
    return report,images


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='current');parser.add_argument('--sky',action='store_true')
    parser.add_argument('--motion-jitter',type=float,default=.25);args=parser.parse_args()
    report,images=measure(not args.sky,args.motion_jitter);out=Path('output/distant-ink');out.mkdir(parents=True,exist_ok=True)
    report['method']='32 post-warmup frames; 480x300 at 32m; rail widths 0.225/0.45/0.9 pixels; slow/fast pans 0.225/2.16 pixels/frame; 4x-per-axis reference; linear SDR errors; analytic rail support with a 3-pixel margin for the no-trails check'
    for name,pixels in images.items():
        pixels=np.clip(pixels,0,1)
        pixels=np.where(pixels<=.0031308,pixels*12.92,1.055*pixels**(1/2.4)-.055)
        Image.fromarray(np.uint8(pixels*255+.5)).resize((960,600),Image.Resampling.NEAREST).save(out/f'{args.name}-{name}.png')
    (out/f'{args.name}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
