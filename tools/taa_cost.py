"""Same-process A/B to separate stationary-cache cost from machine load."""
import json
import time
from pathlib import Path
import numpy as np
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene


def main():
    r=Renderer(load_scene(Path.home()/'Downloads/Bistro_Godot.glb'),1280,800)
    new=r.taa_kernel
    source=(Path(__file__).resolve().parents[1]/'inkforge/shaders/taa.slang').read_text()
    start=source.index('    // Store each of the eight')
    end=source.index('    // Reconstruct the current sample',start)
    module=r.device.load_module_from_source('taa_without_stationary_cache',source[:start]+source[end:])
    old=r.device.create_compute_kernel(r.device.link_program([module],[module.entry_point('taaMain')]))
    results={};s=Style()
    for label,kernel in [('before',old),('after',new)]:
        r.taa_kernel=kernel;r.taa_key=None;results[label]={}
        for mode in ('stationary','moving'):
            samples=[]
            for frame in range(48):
                offset=np.array([frame*.035,0,frame*.025]) if mode=='moving' else np.zeros(3)
                start=time.perf_counter()
                r.render(np.array([-5.,2.2,6])+offset,np.array([2.,4,0])+offset,s)
                r.device.wait()
                if frame>=16:samples.append((time.perf_counter()-start)*1000)
            results[label][mode]=float(np.median(samples))
    Path('output/taa-cost.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
