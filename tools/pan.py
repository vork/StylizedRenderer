"""Render an actual camera translation for temporal style review (no AI images)."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from inkforge.scene import load_scene
from inkforge.renderer import Renderer,preset,Style


def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=60)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=600)
    p.add_argument('--ssaa',type=int,default=2);p.add_argument('--scene',choices=['sponza','bistro'],default='sponza')
    p.add_argument('--aa',choices=['taa','none'],default='taa')
    p.add_argument('--capture',type=Path);p.add_argument('--label');p.add_argument('--baseline',action='store_true')
    a=p.parse_args();out=Path('output')/(a.label or f'{a.scene}-pan');out.mkdir(parents=True,exist_ok=True)
    s=preset('rush');fov=55;cloud_time=0
    if a.capture:
        data=json.loads(a.capture.read_text());scene=load_scene(data['source'])
        eye=np.asarray(data['eye']);target=np.asarray(data['target']);s=Style(**data['style'])
        delta=np.cross(target-eye,[0,1,0]);delta=delta/max(np.linalg.norm(delta),1e-6)*1.5
        fov=data['fov'];cloud_time=data['cloud_time_seconds']
    elif a.scene=='sponza':
        scene=load_scene('assets/Sponza/glTF/Sponza.gltf');eye=np.array([7.,2.3,0]);target=np.array([-5.,4,0]);delta=np.array([0,0,1.5])
    else:
        scene=load_scene(Path.home()/'Downloads/Bistro_Godot.glb');eye=np.array([-5.,2.2,6]);target=np.array([2.,4,0]);delta=np.array([1.5,0,1.5])
    r=Renderer(scene,a.width*a.ssaa,a.height*a.ssaa);r.temporal_coverage=not a.baseline
    s.outline*=a.ssaa;s.bloom_dot_size*=a.ssaa
    s.outline_wobble*=a.ssaa
    s.taa=a.aa=='taa'
    if s.taa:
        for _ in range(32):r.render(eye-delta*.5,target-delta*.5,s,fov,time_seconds=cloud_time)
    frames=[]
    for i in range(a.frames):
        offset=delta*(i/(a.frames-1)-.5)
        r.render(eye+offset,target+offset,s,fov,time_seconds=cloud_time+i/30)
        pixels=r.output.to_numpy()
        assert np.isfinite(pixels).all()
        im=Image.fromarray(np.uint8(np.clip(pixels[:,:,:3],0,1)*255+.5))
        im=im.resize((a.width,a.height),Image.Resampling.LANCZOS)
        im.save(out/f'{i:04d}.png');frames.append(im)
        if i%15==0: print(f'Pan frame {i+1}/{a.frames}',flush=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','30','-i',str(out/'%04d.png'),
                    '-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(out.with_suffix('.mp4'))],check=True)
    # One fixed palette for the GIF avoids palette flicker between frames.
    palette=frames[len(frames)//2].quantize(colors=256)
    gif=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames[::2]]
    gif[0].save(out.with_suffix('.gif'),save_all=True,append_images=gif[1:]+list(reversed(gif[1:-1])),duration=67,loop=0)
    (out/'camera.json').write_text(json.dumps({'eye':eye.tolist(),'target':target.tolist(),'translation':delta.tolist(),
        'frames':a.frames,'ssaa':a.ssaa,'temporal_aa':s.taa,'cloud_animation':True,'coverage_history':r.temporal_coverage},indent=2))
    print(out.with_suffix('.mp4'),flush=True)


if __name__=='__main__':main()
