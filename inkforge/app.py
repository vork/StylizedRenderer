import argparse
import json
from pathlib import Path
from .scene import load_scene
from .renderer import Renderer, Style, default_camera, preset
from .environment import LIGHTING_MODES,GROUND_MODES,with_ground


def main():
    p=argparse.ArgumentParser(description='Inkforge — Slang stylized PBR renderer')
    p.add_argument('scene',type=Path,nargs='?',help='Optional in viewer mode; drop a scene onto the window')
    p.add_argument('--session',type=Path,help='Restore camera and style from a saved capture JSON')
    p.add_argument('--output',type=Path,default=Path('output/render.png'))
    p.add_argument('--style',choices=['rush','graphic','soft','pbr'],default='rush')
    p.add_argument('--backend',choices=['auto','metal','vulkan','d3d12'],default='auto')
    p.add_argument('--width',type=int,default=1280); p.add_argument('--height',type=int,default=720)
    p.add_argument('--ssaa',type=int,choices=[1,2],default=1)
    p.add_argument('--eye',nargs=3,type=float); p.add_argument('--target',nargs=3,type=float)
    p.add_argument('--fov',type=float,default=55)
    p.add_argument('--exposure',type=float);p.add_argument('--ao-radius',type=float)
    p.add_argument('--dots',type=float);p.add_argument('--dot-frequency',type=float)
    p.add_argument('--outline',type=float);p.add_argument('--specular-size',type=float)
    p.add_argument('--outline-wobble',type=float)
    p.add_argument('--aa',choices=['taa','none'],default='taa')
    p.add_argument('--penumbra-strength',type=float)
    p.add_argument('--shadow-fill',type=float,help='Ambient light retained in shadow cores (0–1)')
    p.add_argument('--lighting',choices=LIGHTING_MODES)
    p.add_argument('--time-of-day',type=float)
    p.add_argument('--cloud-shadows',type=float,help='Cloud shadow strength (0–1)')
    p.add_argument('--hatch-density',type=float,help='Surface hatch frequency multiplier')
    p.add_argument('--ground',choices=GROUND_MODES,default='auto')
    p.add_argument('--hatching',type=float);p.add_argument('--palette',type=float)
    p.add_argument('--hatch-direction',type=int,choices=[0,1,2,3],help='Surface UV direction: horizontal, vertical, +45, -45')
    p.add_argument('--bloom',type=float);p.add_argument('--bloom-threshold',type=float);p.add_argument('--bloom-dot-size',type=float)
    p.add_argument('--shadow-distance',type=float);p.add_argument('--shadow-size',type=int,default=2048)
    debug_views=['beauty','ao','normals','albedo','shadows','halftone','hatching','patterns','cascades','cast-shadows','terminator','printed-shadows']
    p.add_argument('--debug',choices=debug_views,default='beauty')
    p.add_argument('--max-texture',type=int,default=1024)
    p.add_argument('--viewer',action='store_true')
    args=p.parse_args()
    session=json.loads(args.session.read_text()) if args.session else None
    if session:
        args.scene=None if session['source']=='procedural material showroom' else Path(session['source'])
        if args.eye is None:args.eye=session['eye']
        if args.target is None:args.target=session['target']
        args.width,args.height=session['render_resolution'];args.fov=session['fov']
    if min(args.width,args.height,args.max_texture)<=0: p.error('Dimensions must be positive')
    if args.scene is None:
        from .procedural import showroom
        scene=showroom();args.viewer=True
    else:scene=load_scene(args.scene)
    eye,target=default_camera(scene)
    if args.eye is not None: eye=args.eye
    if args.target is not None: target=args.target
    style=Style(**session['style']) if session else preset(args.style)
    if args.lighting is not None:style.lighting_mode=args.lighting
    if args.time_of_day is not None:style.time_of_day=args.time_of_day%24
    style.taa=args.aa=='taa'
    if args.penumbra_strength is not None:style.penumbra_strength=args.penumbra_strength
    if args.shadow_fill is not None:style.shadow_fill=args.shadow_fill
    if args.cloud_shadows is not None:style.cloud_shadows=args.cloud_shadows
    if args.hatch_density is not None:style.hatch_density=args.hatch_density
    for arg,attr in [('exposure','exposure'),('ao_radius','ao_radius'),('dots','halftone'),('dot_frequency','dot_frequency'),('outline','outline'),('specular_size','specular_size'),('hatching','hatching'),('palette','palette'),('shadow_distance','shadow_distance'),('bloom','bloom'),('bloom_threshold','bloom_threshold'),('bloom_dot_size','bloom_dot_size')]:
        val=getattr(args,arg)
        if val is not None: setattr(style,attr,val)
    style.outline*=args.ssaa
    if args.outline_wobble is not None:style.outline_wobble=args.outline_wobble
    style.outline_wobble*=args.ssaa
    if args.hatch_direction is not None:style.hatch_direction=args.hatch_direction
    style.bloom_dot_size*=args.ssaa
    r=Renderer(with_ground(scene,args.ground),args.width*args.ssaa,args.height*args.ssaa,args.backend,args.max_texture,args.shadow_size)
    if args.viewer:
        from .viewer import show
        show(r,eye,target,style,args.fov,scene,args.ground,time_seconds=session.get('cloud_time_seconds',0) if session else 0)
    else:
        r.save(args.output,eye,target,style,args.fov,debug_views.index(args.debug),args.ssaa)


if __name__=='__main__': main()
