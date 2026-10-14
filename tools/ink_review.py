"""Real-render evidence for tiny ink, bloom, moving cloud shadows and edge stability."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkforge.procedural import showroom,plane
from inkforge.scene import Scene,Material,Primitive,load_scene
from inkforge.renderer import Renderer,Style


def main():
    out=Path('output/ink-review');out.mkdir(parents=True,exist_ok=True)
    r=Renderer(showroom(),640,400,shadow_size=1024);s=Style(ao_radius=.7)
    raw=[];stable=[];spatial=[];masks=[]
    for frame in range(40):
        r.render([4,4,-7],[0,.5,1],s)
        if frame>=16:
            raw.append(r.mapped.to_numpy()[:,:,:3].copy())
            stable.append(r.history_color[(r.taa_frame-1)%2].to_numpy()[:,:,:3].copy())
            spatial.append(r.antialiased.to_numpy()[:,:,:3].copy())
            masks.append(r.ink_mask.to_numpy().squeeze()>.1)
    edge=np.any(masks,axis=0)
    report={'edge_pixels':int(edge.sum()),'unresolved_edge_temporal_std':float(np.std(np.asarray(raw,dtype=np.float32),axis=0)[edge].mean()),
            'taa_edge_temporal_std':float(np.std(np.asarray(stable,dtype=np.float32),axis=0)[edge].mean()),
            'final_edge_temporal_std':float(np.std(np.asarray(spatial,dtype=np.float32),axis=0)[edge].mean()),
            'method':'24 settled stationary frames, changing projection jitter, union of actual outline masks, linear SDR 0–1; not a moving-camera ghosting metric'}
    v,i=plane(200)
    scene=Scene([Primitive(v,i,0)],[Material(base=[.5,.45,.3,1],metallic=0)],[],'Cloud shadow field')
    r=Renderer(scene,960,600,shadow_size=512);s=Style(fog=.001)
    for stamp in (0,40):r.save(out/f'cloud-shadows-{stamp}.png',[0,100,150],[0,0,0],s,time_seconds=stamp)
    r=Renderer(load_scene(Path.home()/'Downloads/Bistro_Godot.glb'),1280,800);s=Style()
    r.save(out/'bloom-on.png',[-5,2.2,6],[2,4,0],s)
    on=r.output.to_numpy()[:,:,:3].copy()
    s.bloom=0;r.save(out/'bloom-off.png',[-5,2.2,6],[2,4,0],s)
    difference=np.maximum(on-r.output.to_numpy()[:,:,:3],0)
    report['bloom_pixels_over_1_srgb_code']=int((difference.max(2)>1/255).sum())
    report['bloom_max_srgb_delta']=float(difference.max())
    Image.fromarray(np.uint8(np.clip(difference*10,0,1)*255)).save(out/'bloom-difference-10x.png')
    sheet=Image.new('RGB',(1280,500),(22,24,29));draw=ImageDraw.Draw(sheet)
    for index,name in enumerate(('bloom-off','bloom-on')):
        crop=Image.open(out/f'{name}.png').crop((210,285,610,555)).resize((640,432))
        sheet.paste(crop,(index*640,42));draw.text((index*640+16,14),name.replace('-',' ').title(),fill='white')
    sheet.save(out/'bloom-comparison.png')
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
