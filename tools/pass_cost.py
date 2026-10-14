"""Same-process pass bypass timings; frozen outputs isolate approximate pass cost.
Not GPU timestamps: includes submission and synchronization. Do not use skipped
pass frames as visual evidence.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene

class Skip:
    def dispatch(self,**kwargs):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--label',default='before');a=p.parse_args()
    data=json.loads(a.capture.read_text());r=Renderer(load_scene(data['source']),*data['render_resolution']);s=Style(**data['style'])
    groups={'baseline':[], 'gtao':['gtao','denoise','ao_upsample'],'shadow_filter':['shadow_mask'],'lighting':['lighting'],'taa':['taa_kernel','edge_aa'],'bloom':['bloom_extract','bloom_blur','bloom_composite'],'tonemap':['tonemap'],'geometry':[]}
    original={name:getattr(r,name) for group in groups.values() for name in group};raster=r.raster
    results={}
    for name,group in groups.items():
        for key,value in original.items():setattr(r,key,value)
        r.raster=raster
        for _ in range(8):r.render(data['eye'],data['target'],s,data['fov'],time_seconds=data['cloud_time_seconds'])
        r.device.wait()
        for key in group:setattr(r,key,Skip())
        if name=='geometry':r.raster=lambda *args,**kwargs:None
        times=[]
        for frame in range(32):
            start=time.perf_counter();r.render(data['eye'],data['target'],s,data['fov'],time_seconds=data['cloud_time_seconds']);r.device.wait()
            if frame>=8:times.append((time.perf_counter()-start)*1000)
        results[name]=float(np.median(times))
    out=Path('output')/f'pass-cost-{a.label}.json';out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
