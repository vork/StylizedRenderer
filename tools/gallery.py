"""Reproduce review images with explicit cameras and settings."""
import argparse
from pathlib import Path
from inkforge.scene import load_scene
from inkforge.renderer import Renderer,preset
from inkforge.procedural import showroom,export_usd


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['bistro','sponza','materials','helmet','metalrough'],required=True)
    p.add_argument('--ssaa',type=int,default=1);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=800)
    p.add_argument('--bistro',type=Path,default=Path.home()/'Downloads/Bistro_Godot.glb')
    a=p.parse_args();out=Path('output');out.mkdir(exist_ok=True)
    if a.scene=='bistro':
        scene=load_scene(a.bistro)
        cameras=[('cafe',[-5,2.2,6],[2,4,0]),('street',[-17,2.6,-5],[-3,4,-18]),('terrace',[11,2.4,9],[1,3,1])]
    elif a.scene=='sponza':
        scene=load_scene('assets/Sponza/glTF/Sponza.gltf')
        cameras=[('hall',[7,2.3,0],[-5,4,0])]
    elif a.scene=='helmet':
        scene=load_scene('assets/DamagedHelmet/glTF-Binary/DamagedHelmet.glb')
        cameras=[('helmet',[2,.9,3],[0,0,0])]
    elif a.scene=='metalrough':
        scene=load_scene('assets/MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb')
        cameras=[('grid',[0,0,12],[0,0,-1])]
    else:
        Path('assets').mkdir(exist_ok=True)
        export_usd(showroom(),Path('assets/material-showroom.usda'))
        scene=load_scene('assets/material-showroom.usda')
        cameras=[('materials',[4,4,-7],[0,.5,1])]
    r=Renderer(scene,a.width*a.ssaa,a.height*a.ssaa)
    for name,eye,target in cameras:
        for mode in (['rush'] if len(cameras)>1 else ['rush','graphic','pbr']):
            style=preset(mode)
            style.outline*=a.ssaa
            style.outline_wobble*=a.ssaa
            style.bloom_dot_size*=a.ssaa
            if a.scene=='materials':style.ao_radius=.7
            if a.scene=='helmet':style.dot_frequency=18;style.ao_radius=.3
            r.save(out/f'{a.scene}-{name}-{mode}.png',eye,target,style,downsample=a.ssaa)
        if len(cameras)==1 or name=='cafe':
            r.save(out/f'{a.scene}-{name}-gtao.png',eye,target,preset('rush'),debug=1,downsample=a.ssaa)
            r.save(out/f'{a.scene}-{name}-shadows.png',eye,target,preset('rush'),debug=4,downsample=a.ssaa)
            r.save(out/f'{a.scene}-{name}-cascades.png',eye,target,preset('rush'),debug=8,downsample=a.ssaa)


if __name__=='__main__':main()
