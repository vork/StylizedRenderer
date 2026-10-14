"""Same-scene correctness and wall-time comparison of Metal geometry paths."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args()
    data=json.loads(a.capture.read_text());r=Renderer(load_scene(data['source']),*data['render_resolution'])
    s=Style(**data['style']);s.taa=False;results={};reference=None
    for label,enabled in [('five_pass',False),('depth_and_storage',True)]:
        r.storage_geometry=enabled;samples=[]
        for frame in range(32):
            start=time.perf_counter()
            r.render(data['eye'],data['target'],s,data['fov'],time_seconds=data['cloud_time_seconds'])
            r.device.wait()
            if frame>=8:samples.append((time.perf_counter()-start)*1000)
        current=r.output.to_numpy().copy()
        if reference is None:reference=current
        results[label]={'median_frame_ms':float(np.median(samples)),
                        'image_mean_error':float(abs(current-reference).mean()),
                        'pixels_over_1_code':int((abs(current-reference)[:,:,:3].max(2)>1/255).sum())}
    out=Path('output/geometry-review.json');out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
