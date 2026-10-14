"""Matched-camera A/B of distant outline variation, with light and clouds held fixed."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path)
    p.add_argument('--stationary',action='store_true',help='Compare the exact stationary resolve with the previous moving-camera TAA')
    a=p.parse_args()
    data=json.loads(a.capture.read_text());s=Style(**data['style'])
    r=Renderer(load_scene(data['source']),*data['render_resolution'])
    out=Path('output')/('stationary-review' if a.stationary else 'distant-ink');out.mkdir(parents=True,exist_ok=True);report={}
    for label,enabled in [('bistro-before',False),('bistro-after',True)]:
        if a.stationary:r.stationary_resolve=enabled
        else:r.temporal_coverage=enabled;r.stationary_resolve=False
        frames=[];masks=[];final_frames=[]
        for frame in range(64):
            r.render(data['eye'],data['target'],s,data['fov'],time_seconds=data['cloud_time_seconds'])
            if frame>=32:
                color=r.antialiased.to_numpy()[:,:,:3].astype(np.float32)
                frames.append(color@np.array([.2126,.7152,.0722],np.float32))
                final=r.output.to_numpy()[:,:,:3].astype(np.float32)
                final_frames.append(final@np.array([.2126,.7152,.0722],np.float32))
                world=r.gbuffers[0].to_numpy()
                far=(world[:,:,3]!=0)&(np.linalg.norm(world[:,:,:3]-data['eye'],axis=2)>24)
                masks.append(far&(r.ink_mask.to_numpy().squeeze()>.05))
        mask=np.any(masks,axis=0);std=np.std(frames,axis=0)
        report[label]={'far_ink_pixels':int(mask.sum()),'temporal_std_mean':float(std[mask].mean()),
                       'temporal_std_p95':float(np.percentile(std[mask],95)),
                       'whole_frame_srgb_std_mean':float(np.std(final_frames,axis=0).mean()),
                       'whole_frame_srgb_std_p99':float(np.percentile(np.std(final_frames,axis=0),99))}
        image=np.uint8(np.clip(r.output.to_numpy()[:,:,:3],0,1)*255+.5)
        Image.fromarray(image).save(out/f'{label}.png')
        Image.fromarray(np.uint8(np.clip(std*10,0,1)*255)).save(out/f'{label}-variation-10x.png')
    report['method']='32 settled frames after 32 warmup; frozen camera, lighting, clouds; actual saved resolution; union of outline pixels on geometry beyond 24m; display-linear luminance'
    report['capture']=str(a.capture);report['stationary_resolve_comparison']=a.stationary
    (out/'bistro-temporal.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
