"""Warm, synchronized frame timings. No image readback, PNG encoding, or vsync."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import slangpy as spy
from inkforge.renderer import Renderer,preset
from inkforge.scene import load_scene


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['bistro','sponza'],required=True)
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=800)
    p.add_argument('--frames',type=int,default=40)
    p.add_argument('--label',default='cached')
    p.add_argument('--bloom',type=float,default=.04)
    p.add_argument('--aa',choices=['taa','none'],default='taa')
    a=p.parse_args()
    if a.scene=='bistro':
        scene=load_scene(Path.home()/'Downloads/Bistro_Godot.glb')
        eye=np.array([-5.,2.2,6]);target=np.array([2.,4,0])
    else:
        scene=load_scene('assets/Sponza/glTF/Sponza.gltf')
        eye=np.array([7.,2.3,0]);target=np.array([-5.,4,0])
    r=Renderer(scene,a.width,a.height);s=preset('rush');s.bloom=a.bloom
    s.taa=a.aa=='taa'
    report={'gpu':r.device.info.adapter_name,'backend':r.backend,
            'scene':a.scene,'triangles':scene.triangle_count,'batches':len(r.meshes),
            'resolution':[a.width,a.height],'shadow_maps':'4 x 2048',
            'anti_aliasing':a.aa,'shadow_fill':s.shadow_fill,'penumbra_strength':s.penumbra_strength,
            'bloom':a.bloom,'shadow_style':'solid cores, stippled penumbras, surface UV AO tonal maps',
            'frames':a.frames,'warmup':8,'gpu_timestamp_support':r.device.has_feature(spy.Feature.timestamp_query),
            'method':'CPU encoding/submission and synchronized full-frame wall time; excludes readback, presentation, loading and shader compilation. GPU and CPU serialized per frame.'}
    for mode in ('stationary','moving','cycling'):
        moving=mode=='moving'
        total=[];cpu=[]
        for frame in range(a.frames+8):
            offset=np.array([frame*.035,0,frame*.025]) if moving else np.zeros(3)
            start=time.perf_counter()
            s.time_of_day=10.+frame/30*.02 if mode=='cycling' else 10.
            r.render(eye+offset,target+offset,s,time_seconds=frame/30 if mode=='cycling' else 0.)
            submitted=time.perf_counter()
            r.device.wait()
            done=time.perf_counter()
            if frame>=8:
                total.append((done-start)*1000);cpu.append((submitted-start)*1000)
        def stats(values):
            return {key:round(float(value),3) for key,value in zip(['mean_ms','p50_ms','p95_ms'],[np.mean(values),*np.percentile(values,[50,95])])}
        report[mode]={'full_frame':stats(total),'cpu_encode_submit':stats(cpu),'equivalent_fps':round(1000/np.mean(total),1)}
    out=Path('output')/f'performance-{a.scene}-{a.width}x{a.height}-{a.label}.json'
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
