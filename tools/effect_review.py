"""Matched beauty/effect-off renders; quantify visible contribution after AA."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene
from inkforge.procedural import showroom


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--label',default='before');a=p.parse_args()
    data=json.loads(a.capture.read_text())
    scene=showroom() if data['source']=='procedural material showroom' else load_scene(data['source'])
    r=Renderer(scene,*data['render_resolution'])
    out=Path('output/effect-review');out.mkdir(parents=True,exist_ok=True);report={}
    beauty=None
    for name,field in [('beauty',None),('ao','ao'),('bloom','bloom'),('stipple','penumbra_strength')]:
        s=Style(**data['style'])
        if field:setattr(s,field,0.)
        r.save(out/f'{a.label}-{name}.png',data['eye'],data['target'],s,data['fov'],time_seconds=data['cloud_time_seconds'])
        pixels=r.output.to_numpy()[:,:,:3].copy()
        if beauty is None:beauty=pixels;continue
        delta=abs(pixels-beauty).max(axis=2)
        report[name]={'pixels_over_1_code':int((delta>1/255).sum()),'mean_delta':float(delta.mean()),'p99_delta':float(np.percentile(delta,99)),'max_delta':float(delta.max())}
        Image.fromarray(np.uint8(np.clip(delta*8,0,1)*255)).save(out/f'{a.label}-{name}-difference-8x.png')
    report['capture']=str(a.capture)
    (out/f'{a.label}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
