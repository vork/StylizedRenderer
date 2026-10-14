import json
import numpy as np
import pytest
from inkforge.scene import Scene,Material,Primitive,Light,load_scene
from inkforge.procedural import sphere,showroom,export_usd
from inkforge.environment import lighting_state,with_ground,needs_ground
from inkforge.renderer import Style,Renderer
from inkforge.viewer_state import dropped_scene_path,load_dropped_scene


def object_scene():
    v,i=sphere(np.zeros(3))
    return Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Object')


def test_ground_for_object_without_duplication():
    scene=object_scene();original=scene.triangle_count
    grounded=with_ground(scene)
    assert scene.triangle_count==original
    assert grounded.triangle_count==original+2
    assert grounded.bounds[0][1]<scene.bounds[0][1]
    assert not needs_ground(grounded)
    assert with_ground(grounded) is grounded
    assert with_ground(scene,'off') is scene
    assert not needs_ground(showroom())


def test_lighting_modes_preserve_or_replace_authored_lights():
    scene=object_scene();s=Style()
    automatic=lighting_state(scene,s)
    assert automatic['power']>0 and automatic['procedural']
    local=Light(kind='point',position=(1,2,3),intensity=20)
    scene.lights=[local]
    assert lighting_state(scene,s)['keep'] and lighting_state(scene,s)['power']>0
    scene.lights.append(Light(kind='directional',direction=(0,-1,0),intensity=3))
    state=lighting_state(scene,s)
    assert state['skip']==1 and state['power']==3
    assert state['cloud_key']
    s.lighting_mode='replace';state=lighting_state(scene,s)
    assert not state['keep'] and state['skip']==-1
    s.lighting_mode='add';state=lighting_state(scene,s)
    assert state['keep'] and state['skip']==-1 and state['power']>0
    s.lighting_mode='sky_only';state=lighting_state(scene,s)
    assert state['keep'] and state['power']==3
    scene.lights=[local]
    assert lighting_state(scene,s)['power']==0
    scene.lights=[Light(kind='dome',color=(.3,.4,.5),intensity=2)]
    s.lighting_mode='auto';state=lighting_state(scene,s)
    assert not state['procedural'] and state['power']==0
    np.testing.assert_allclose(state['sky'],[.6,.8,1.])


def test_day_night_lighting_tracks_celestial_direction():
    s=Style();scene=object_scene()
    s.time_of_day=12;day=lighting_state(scene,s)
    s.time_of_day=0;night=lighting_state(scene,s)
    assert day['sun_direction'][1]>.99 and night['sun_direction'][1]<-.99
    assert night['direction'][1]>.99 and 0<night['power']<day['power']*.1
    assert np.mean(night['sky'])<np.mean(day['sky'])*.1


def test_usd_imports_visible_transformed_lights(tmp_path):
    from pxr import Usd,UsdLux,UsdGeom,Gf
    path=tmp_path/'lights.usda';export_usd(object_scene(),path)
    stage=Usd.Stage.Open(str(path))
    sun=UsdLux.DistantLight.Define(stage,'/Sun');sun.CreateIntensityAttr(3.)
    lamp=UsdLux.SphereLight.Define(stage,'/Lamp');lamp.CreateIntensityAttr(10.)
    lamp.CreateNormalizeAttr(True);UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    shape=UsdLux.ShapingAPI.Apply(lamp.GetPrim());shape.CreateShapingConeAngleAttr(30.)
    shape.CreateShapingConeSoftnessAttr(.25)
    hidden=UsdLux.SphereLight.Define(stage,'/Hidden');UsdGeom.Imageable(hidden).MakeInvisible()
    stage.GetRootLayer().Save()
    scene=load_scene(path)
    assert len(scene.lights)==2
    assert scene.lights[0].kind=='directional' and scene.lights[0].intensity==3
    light=scene.lights[1]
    assert light.kind=='spot' and light.intensity==10
    np.testing.assert_allclose(light.position,[1,2,3])
    assert abs(light.outer-np.pi/6)<1e-5


def test_drop_validation_and_device_reuse(tmp_path):
    path=tmp_path/'object.usda';export_usd(object_scene(),path)
    assert dropped_scene_path([str(tmp_path/'missing.glb'),str(path)])==path
    with pytest.raises(ValueError):dropped_scene_path(['invalid.txt'])
    old=Renderer(showroom(),96,64,shadow_size=128)
    new,raw=load_dropped_scene(old,[str(path)],'auto')
    assert new.device is old.device and new.scene.triangle_count==raw.triangle_count+2
    new.render([2,1,3],[0,0,0],Style())
    assert np.isfinite(new.output.to_numpy()).all()
    broken=tmp_path/'bad.glb';broken.write_bytes(b'not a glb')
    with pytest.raises(Exception):load_dropped_scene(old,[str(broken)],'auto')
    old.render([4,4,-7],[0,.5,1],Style())
    assert np.isfinite(old.output.to_numpy()).all()


def test_local_lights_affect_pixels_and_replace_mode_ignores_them():
    scene=with_ground(object_scene());scene.lights=[Light(position=(1,2,2),color=(1,.05,.01),intensity=35)]
    r=Renderer(scene,160,100,shadow_size=256);s=Style(taa=False,lighting_mode='sky_only',bloom=0)
    r.render([2,1,3],[0,0,0],s);lit=r.hdr.to_numpy().copy()
    s.lighting_mode='replace';r.render([2,1,3],[0,0,0],s);replacement=r.hdr.to_numpy().copy()
    assert np.mean(abs(lit-replacement))>.02
    # Same geometry/device, remove scene lights from the dispatch while replacing.
    scene.lights=[]
    r.render([2,1,3],[0,0,0],s)
    np.testing.assert_allclose(r.hdr.to_numpy(),replacement,atol=.001)


def test_sky_clouds_animate_and_night_has_stars():
    r=Renderer(object_scene(),256,160,shadow_size=128);s=Style(taa=False,bloom=0)
    eye=[0,2,4];target=[0,5,-4]
    r.render(eye,target,s,time_seconds=0);day=r.hdr.to_numpy().copy()
    r.render(eye,target,s,time_seconds=60);moved=r.hdr.to_numpy().copy()
    assert np.mean(abs(day-moved))>.005
    s.time_of_day=0;r.render(eye,target,s);night=r.hdr.to_numpy().copy()
    assert np.isfinite(night).all()
    assert np.mean(night[:,:,:3])<np.mean(day[:,:,:3])*.4
    assert (night[:,:,:3].max(2)>.5).sum()>3


@pytest.mark.parametrize('authored_key',[False,True])
def test_cloud_shadows_move_on_geometry_and_can_be_disabled(authored_key):
    from inkforge.procedural import plane
    v,i=plane(250)
    scene=Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Cloud shadow receiver')
    if authored_key:
        scene.lights=[Light(kind='directional',direction=(.3,-1,.2),intensity=4)]
    r=Renderer(scene,256,160,shadow_size=128)
    s=Style(taa=False,bloom=0,cloud_shadows=.7)
    eye=[0,80,120];target=[0,0,0]
    r.render(eye,target,s,time_seconds=0);first=r.hdr.to_numpy().copy()
    r.render(eye,target,s,time_seconds=70);second=r.hdr.to_numpy().copy()
    mask=r.gbuffers[0].to_numpy()[:,:,3]!=0
    assert np.mean(abs(first[mask,:3]-second[mask,:3]))>.01
    s.cloud_shadows=0
    r.render(eye,target,s,time_seconds=0);first=r.hdr.to_numpy().copy()
    r.render(eye,target,s,time_seconds=70);second=r.hdr.to_numpy().copy()
    np.testing.assert_allclose(first[mask,:3],second[mask,:3],atol=1e-5)
