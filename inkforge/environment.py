"""Scene lighting policy and optional object ground. No GPU dependencies."""
import numpy as np
from .math3d import normalize
from .scene import Scene,Material,Primitive,make_vertices

LIGHTING_MODES=('auto','replace','add','sky_only')
GROUND_MODES=('auto','on','off')


def lighting_state(scene,style):
    if style.lighting_mode not in LIGHTING_MODES:raise ValueError('Unknown lighting mode')
    phase=(style.time_of_day-6)*np.pi/12
    sun=normalize([-np.cos(phase)*.85,np.sin(phase)*.9,np.cos(phase)*.55])
    daylight=float(np.clip((sun[1]+.08)/.3,0,1));daylight=daylight*daylight*(3-2*daylight)
    day_color=np.array([1.,.42,.18])*(1-np.clip(sun[1]*2,0,1))+np.array(style.sun_color)*np.clip(sun[1]*2,0,1)
    celestial_direction=sun if sun[1]>=0 else -sun
    celestial_color=day_color if sun[1]>=0 else np.array([.48,.6,1.])
    horizon=float(np.clip(abs(sun[1])/.12,0,1));horizon=horizon*horizon*(3-2*horizon)
    celestial_power=style.sun*(daylight if sun[1]>=0 else .035*(1-daylight))*horizon
    keep=style.lighting_mode!='replace'
    directional=[(i,l) for i,l in enumerate(scene.lights) if l.kind=='directional' and l.intensity>0]
    dome=[l for l in scene.lights if l.kind=='dome' and l.intensity>0]
    add_key=style.lighting_mode in ('replace','add') or (style.lighting_mode=='auto' and not directional and not dome)
    primary=max(directional,key=lambda x:x[1].intensity) if keep and directional and not add_key else None
    direction=celestial_direction;power=celestial_power if add_key else 0.;color=celestial_color;skip=-1
    if primary:
        skip,light=primary;direction=-normalize(light.direction);power=light.intensity;color=np.array(light.color)
    procedural=style.lighting_mode!='auto' or not dome
    sky=np.array(style.sky)*(.035+.965*daylight)
    tint=np.array(style.shadow_tint)*(.06+.94*daylight)
    if not procedural:
        sky=sum((np.array(l.color)*l.intensity for l in dome),start=np.zeros(3));tint=sky*.6
    return dict(direction=direction,color=color,power=float(power),sun_direction=sun,daylight=daylight,
                sky=sky,tint=tint,procedural=procedural,keep=keep,skip=skip,cloud_key=procedural and power>0,
                shadow_presence=horizon if add_key else float(power>0))


def pack_lights(lights):
    # Four float4 values per light, matching SceneLight in lighting.slang.
    data=np.zeros((max(1,len(lights)),4,4),np.float32)
    for i,l in enumerate(lights):
        data[i,0]=[*l.position,{'directional':0,'point':1,'spot':2,'dome':3}[l.kind]]
        data[i,1]=[*l.direction,l.range]
        data[i,2]=[*l.color,l.intensity]
        data[i,3]=[np.cos(l.inner),np.cos(l.outer),0,0]
    return data


def needs_ground(scene):
    low,high=scene.bounds;extent=high-low
    # A conservative metre-space object heuristic; G / the selector overrides it.
    if extent.max()>12 or len(scene.primitives)>32:return False
    footprint=max(float(extent[0]*extent[2]),1e-5);floor_area=0.
    for primitive in scene.primitives:
        triangles=primitive.vertices[primitive.indices.reshape(-1,3),:3]
        near=np.max(abs(triangles[:,:,1]-low[1]),axis=1)<max(extent.max()*.01,.001)
        selected=triangles[near]
        if len(selected):
            cross=np.cross(selected[:,1]-selected[:,0],selected[:,2]-selected[:,0])
            floor_area+=np.abs(cross[:,1]).sum()*.5
    # A closed object's bottom can cover its footprint too. A broad flat surface
    # with little vertical extent relative to the footprint is treated as a floor.
    has_floor=floor_area>footprint*.65 and min(extent[0],extent[2])>extent[1]*3
    return not has_floor


def with_ground(scene,mode='auto'):
    if mode not in GROUND_MODES:raise ValueError('Unknown ground mode')
    if mode=='off' or (mode=='auto' and not needs_ground(scene)):return scene
    low,high=scene.bounds;center=(low+high)*.5
    radius=max(float(np.max(high-low))*1.8,.5);y=float(low[1])-.002*max(radius,1)
    x,z=center[0],center[2]
    v,i=make_vertices([[x-radius,y,z-radius],[x-radius,y,z+radius],[x+radius,y,z+radius],[x+radius,y,z-radius]],[0,1,2,0,2,3])
    material=Material(name='Procedural paper ground',base=[.65,.58,.42,1],metallic=0,roughness=.9)
    return Scene(scene.primitives+[Primitive(v,i,len(scene.materials))],scene.materials+[material],scene.images,
                 scene.source,scene.lights).batch()
