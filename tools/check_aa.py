"""Compare real GPU silhouette coverage to a 4x-per-axis reference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from inkforge.scene import Scene,Material,Primitive,make_vertices
from inkforge.renderer import Renderer,Style


def measure(outlined=False):
    v,i=make_vertices([[-1.45,-.8,0],[1.3,-.62,0],[-.27,1.1,0]],[0,1,2])
    scene=Scene([Primitive(v,i,0)],[Material(base=[.04,.015,.008,1],unlit=True,double_sided=True)],[],'AA diagonal silhouette')
    style=Style(amount=float(outlined),taa=False,bloom=0,fog=0,outline_wobble=0)
    eye=[0,0,4];target=[0,0,0]
    reference_renderer=Renderer(scene,512,320,shadow_size=128)
    if outlined:style.outline=4
    reference_renderer.render(eye,target,style)
    style.outline=1
    # Coverage must be averaged in display-linear light, matching the resolve.
    # Averaging sRGB codes makes the reference too dark at high-contrast edges.
    high=reference_renderer.mapped.to_numpy()[:,:,:3].astype(np.float32)
    high_coverage=(reference_renderer.gbuffers[0].to_numpy()[:,:,3]!=0).astype(float)
    coverage=high_coverage.reshape(80,4,128,4).mean(axis=(1,3))
    reference=high.reshape(80,4,128,4,3).mean(axis=(1,3))
    reference=np.where(reference<=.0031308,reference*12.92,1.055*np.maximum(reference,0)**(1/2.4)-.055)
    r=Renderer(scene,128,80,shadow_size=128)
    r.render(eye,target,style);raw=r.output.to_numpy()[:,:,:3].copy()
    style.taa=True
    for _ in range(16):r.render(eye,target,style)
    resolved=r.output.to_numpy()[:,:,:3].copy()
    temporal=r.history_color[(r.taa_frame-1)%2].to_numpy()[:,:,:3].astype(np.float32)
    temporal=np.where(temporal<=.0031308,temporal*12.92,1.055*np.maximum(temporal,0)**(1/2.4)-.055)
    # Only pixels neighboring a silhouette contribute to this measurement.
    edges=abs(coverage-np.roll(coverage,1,axis=0))>.01
    edges|=abs(coverage-np.roll(coverage,1,axis=1))>.01
    mask=edges.copy()
    for axis in (0,1):
        mask|=np.roll(edges,1,axis=axis)|np.roll(edges,-1,axis=axis)
    report={'edge_pixels':int(mask.sum()),'raw_edge_mae':float(abs(raw-reference)[mask].mean()),
            'taa_edge_mae':float(abs(resolved-reference)[mask].mean()),
            'temporal_only_edge_mae':float(abs(temporal-reference)[mask].mean())}
    # A camera cut away from geometry must not leave old silhouettes behind.
    r.render([10,0,4],[10,0,0],style)
    assert r.taa_frame==1
    cut=r.history_color[0].to_numpy()[:,:,:3]
    report['camera_cut_error']=float(abs(cut-r.mapped.to_numpy()[:,:,:3]).max())
    style.exposure+=.5
    r.render(eye,target,style)
    assert r.taa_frame==1
    report['style_reset_error']=float(abs(r.history_color[0].to_numpy()[:,:,:3]-r.mapped.to_numpy()[:,:,:3]).max())
    return report,(raw,resolved,reference)


if __name__=='__main__':
    report,images=measure();out=Path('output');out.mkdir(exist_ok=True)
    for name,im in zip(['none','taa','reference'],images):
        Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).resize((768,480),Image.Resampling.NEAREST).save(out/f'aa-{name}.png')
    (out/'aa-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
