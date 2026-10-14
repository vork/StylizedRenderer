from pathlib import Path
import platform
import numpy as np
import pytest
import slangpy as spy
from inkforge.renderer import Renderer,Style
from inkforge.scene import Scene,Material,Primitive
from inkforge.procedural import plane,showroom

ROOT=Path(__file__).resolve().parents[1]


def test_gt7_matches_polyphony_reference():
    backend='metal' if platform.system()=='Darwin' else 'd3d12' if platform.system()=='Windows' else 'vulkan'
    d=spy.Device(type=getattr(spy.DeviceType,backend),compiler_options={'include_paths':[str(ROOT/'inkforge/shaders'),str(ROOT/'tests')]})
    values=np.array([[0,0,0],[.18,.18,.18],[1,1,1],[2.5,2.5,2.5],[100,100,100],
                     [.5,1.23,.75],[12.3,34.3,56.9],[1504.7,64.51,.5],[10,0,0],[0,10,0],[0,0,10],[.001,.001,.001]],np.float32)
    packed=np.pad(values,((0,0),(0,1)))
    source=d.create_buffer(data=packed,usage=spy.BufferUsage.shader_resource)
    result=d.create_buffer(size=packed.nbytes,usage=spy.BufferUsage.unordered_access)
    kernel=d.create_compute_kernel(d.load_program('verify.slang',['toneTest']))
    kernel.dispatch(thread_count=[len(values),1,1],input=source,result=result,count=len(values))
    actual=result.to_numpy().view(np.float32).reshape(-1,4)[:,:3]
    expected=np.loadtxt(ROOT/'tests/gt7_expected.txt')
    np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=1e-3)


@pytest.fixture(scope='module')
def flat_renderer():
    v,i=plane(20)
    scene=Scene([Primitive(v,i,0)],[Material(base=[.5,.5,.5,1],metallic=0)],[],'AO validation plane')
    return Renderer(scene,256,160,shadow_size=512)


@pytest.mark.parametrize('eye',[[0,4,5],[0,1,6],[5,2,2]])
def test_gtao_unoccluded_plane_is_bright(flat_renderer,eye):
    r=flat_renderer;r.render(eye,[0,0,0],Style())
    pos=r.gbuffers[0].to_numpy();ao=r.ao.to_numpy().squeeze()
    mask=pos[:,:,3]>.5
    assert mask.sum()>1000
    assert np.mean(ao[mask])>.97
    assert np.percentile(ao[mask],5)>.9
    pixels=r.output.to_numpy()
    assert np.isfinite(pixels).all() and pixels[:,:,:3].mean()>.05


def test_shadow_plane_has_no_normal_dependent_acne(flat_renderer):
    r=flat_renderer;r.render([0,2,5],[0,0,0],Style(),debug=4)
    p=r.gbuffers[0].to_numpy();mask=p[:,:,3]>.5
    colors=r.output.to_numpy()[:,:,0]
    assert np.mean(colors[mask]>.98)>.995


def test_flat_surface_with_smoothed_normals_has_no_shadow_acne():
    # A flat polygon adjoining rounded bevels often has non-planar vertex normals.
    # Shadow depth correction must use its triangle plane, not these normals.
    v,i=plane(20)
    v[:,3:6]=[.35,.9,.26]
    v[:,3:6]/=np.linalg.norm(v[:,3:6],axis=1,keepdims=True)
    scene=Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Smoothed flat receiver')
    r=Renderer(scene,256,160,shadow_size=512)
    for eye in ([0,2,5],[5,2,2]):
        r.render(eye,[0,0,0],Style(),debug=4)
        p=r.gbuffers[0].to_numpy();mask=p[:,:,3]>.5
        assert np.mean(r.output.to_numpy()[:,:,0][mask]>.98)>.995


def bent_receiver_scene(amplitude=.012,grid=97):
    # This height field's maximum slope is below the sun elevation: every point
    # can see the light. Filtering adjacent triangle planes must not add folds
    # of false shadow, even with relatively coarse shadow texels.
    from inkforge.scene import make_vertices
    x,z=np.meshgrid(np.linspace(-2,2,grid),np.linspace(-2,2,grid))
    y=amplitude*np.sin(x*12)*np.sin(z*12)
    points=np.stack([x,y,z],axis=-1).reshape(-1,3)
    normals=np.stack([-amplitude*12*np.cos(x*12)*np.sin(z*12),np.ones_like(x),
                      -amplitude*12*np.sin(x*12)*np.cos(z*12)],axis=-1).reshape(-1,3)
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    ids=np.arange(grid*grid).reshape(grid,grid);a=ids[:-1,:-1].ravel()
    indices=np.stack([a,a+grid,a+1,a+1,a+grid,a+grid+1],axis=1).ravel()
    vertices,indices=make_vertices(points,indices,normals=normals)
    return Scene([Primitive(vertices,indices,0)],[Material(metallic=0)],[],'Bent receiver')


@pytest.mark.parametrize('amplitude,grid',[(.012,97),(.03,33)])
def test_gently_bent_receiver_does_not_shadow_its_neighboring_facets(amplitude,grid):
    scene=bent_receiver_scene(amplitude,grid)
    r=Renderer(scene,320,200,shadow_size=512)
    r.render([0,2,4],[0,0,0],Style(),debug=9)
    p=r.gbuffers[0].to_numpy();mask=(p[:,:,3]!=0)&(abs(p[:,:,0])<1.6)&(abs(p[:,:,2])<1.6)
    assert mask.sum()>1000
    srgb=r.output.to_numpy()[:,:,0][mask]
    visibility=((srgb+.055)/1.055)**2.4
    # Permit sub-percent integrated PCF error, but no invented hard shadows.
    assert visibility.mean()>.995
    assert visibility.min()>.8
    r.render([0,2,4],[0,0,0],Style(),debug=11)
    assert np.mean(r.output.to_numpy()[:,:,0][mask]>.98)>.99


def test_gtao_detects_contacts():
    r=Renderer(showroom(),320,200,shadow_size=512)
    r.render([3,4,-7],[0,.4,1],Style())
    p=r.gbuffers[0].to_numpy();ao=r.ao.to_numpy().squeeze()
    floor=(p[:,:,3]>.5)&(abs(p[:,:,1])<.01)
    assert np.percentile(ao[floor],2)<.8
    assert np.percentile(ao[floor],90)>.95


def test_caster_plane_correction_preserves_close_blockers():
    from inkforge.environment import lighting_state
    from inkforge.scene import make_vertices
    ground,indices=plane(3)
    # A separate tilted sheet only 4 cm above the receiver, similar normals.
    blocker,bi=make_vertices([[-.7,.026,-.7],[-.7,.026,.7],[.7,.054,.7],[.7,.054,-.7]],
                             [0,1,2,0,2,3])
    scene=Scene([Primitive(ground,indices,0),Primitive(blocker,bi,0)],
                [Material(metallic=0,double_sided=True)],[],'Nearby tilted blocker')
    r=Renderer(scene,320,200,shadow_size=512);s=Style()
    # View the underside so the ground's shadow is visible through no geometry.
    r.render([0,-2,3],[0,0,0],s,debug=9)
    p=r.gbuffers[0].to_numpy();l=lighting_state(scene,s)['direction']
    t=(.04+.02*p[:,:,0]-p[:,:,1])/(l[1]-.02*l[0])
    hit=p[:,:,:3]+t[:,:,None]*l
    floor=(p[:,:,3]!=0)&(abs(p[:,:,1])<1e-4)
    core=floor&(abs(hit[:,:,0])<.45)&(abs(hit[:,:,2])<.45)
    lit=floor&((abs(hit[:,:,0])>1)|(abs(hit[:,:,2])>1))
    assert core.sum()>200 and lit.sum()>1000
    assert np.mean(r.output.to_numpy()[:,:,0][core]<.05)>.99
    assert np.mean(r.output.to_numpy()[:,:,0][lit]>.98)>.99


def test_stipple_is_visible_only_near_edges_and_anchored_under_pan(flat_renderer):
    d=flat_renderer.device
    kernel=d.create_compute_kernel(d.load_program(str(ROOT/'tests/penumbra_probe.slang'),['penumbraTest']))
    target=d.create_texture(format=spy.Format.r32_float,width=256,height=128,
        usage=spy.TextureUsage.unordered_access|spy.TextureUsage.shader_resource)
    def render(strength,pan=(0,0)):
        kernel.dispatch(thread_count=[256,128,1],vars=dict(result=target,strength=strength,pan=spy.float2(pan)))
        return target.to_numpy().squeeze().copy()
    core=render(0);printed=render(.55);shifted=render(.55,(7,3))
    assert np.array_equal(printed[:,:45],core[:,:45])
    assert np.array_equal(printed[:,210:],core[:,210:])
    # Fine antialiased marks have less per-pixel variance than the former large
    # dots, but must still modulate the penumbra rather than become a flat wash.
    assert printed[:,118:138].std(axis=0).mean()>.08
    assert np.mean(abs(printed[:,100:155]-core[:,100:155]))>.08
    np.testing.assert_allclose(shifted[:-3,:-7],printed[3:,7:],atol=1e-5)


def test_stipples_remain_pixel_sized_through_magnification(flat_renderer):
    d=flat_renderer.device
    kernel=d.create_compute_kernel(d.load_program(str(ROOT/'tests/penumbra_probe.slang'),['magnifiedStippleTest']))
    result=d.create_texture(format=spy.Format.r32_float,width=256,height=128,
        usage=spy.TextureUsage.unordered_access|spy.TextureUsage.shader_resource)
    for scale in (192.,768.,3072.,12288.):
        kernel.dispatch(thread_count=[256,128,1],vars=dict(result=result,pixelsPerUV=scale))
        pixels=result.to_numpy().squeeze()
        assert pixels.std()>.04
        # A 64x closeup must not turn individual pen marks into 20-pixel blobs.
        for rows in (pixels,pixels.T):
            for row in rows:
                occupied=np.r_[False,row>.5,False].astype(np.int32)
                starts=np.flatnonzero(np.diff(occupied)==1);ends=np.flatnonzero(np.diff(occupied)==-1)
                assert np.max(ends-starts,initial=0)<=3


def test_bloom_is_patterned_and_density_tracks_energy(flat_renderer):
    r=flat_renderer;d=r.device
    black=d.create_texture(format=spy.Format.rgba32_float,width=256,height=160,
        usage=spy.TextureUsage.shader_resource,data=np.zeros((160,256,4),np.float32))
    fractions=[];sites=[];areas=[]
    y,x=np.mgrid[:160,:256]
    cell_x=np.floor((x+y+1)*.70710678/12).astype(int)
    cell_y=np.floor((y-x)*.70710678/12).astype(int)
    cell_ids=cell_x+cell_y*128
    for energy in [0.,.15,1.5]:
        source=d.create_texture(format=spy.Format.rgba32_float,width=64,height=40,
            usage=spy.TextureUsage.shader_resource,data=np.full((40,64,4),energy,np.float32))
        r.bloom_composite.dispatch(thread_count=[256,160,1],vars=dict(hdrSource=black,baseSource=black,source=source,
            linearSampler=r.bloom_sampler,finalImage=r.output,size=spy.uint2(256,160),
            bloomSize=spy.uint2(64,40),exposure=1.,threshold=2.5,strength=1.,dotSize=12.))
        output=r.output.to_numpy()[:,:,0]
        assert np.isfinite(output).all()
        fractions.append(float(np.mean(output>.05)))
        active=output>.01
        occupied=len(np.unique(cell_ids[active]))
        sites.append(occupied);areas.append(float(active.sum()/max(occupied,1)))
        if energy>0:
            assert np.mean(output<.01)>.2  # distinct gaps, not a continuous glow
            assert .05<output.max()<.8  # visible but bounded light accents
    assert fractions[0]==0
    assert 0<fractions[1]<fractions[2]<.8
    assert 0<sites[1]<sites[2]  # fading removes dots, increasing effective spacing
    assert areas[1]<areas[2]   # remaining dots also shrink as the halo fades


def test_bloom_extraction_bounds_extreme_emissive_energy(flat_renderer):
    r=flat_renderer;d=r.device;peaks=[]
    for energy in (1.,3.,100.,1e6):
        data=np.full((160,256,4),energy,np.float32);data[:,:,3]=1
        source=d.create_texture(format=spy.Format.rgba32_float,width=256,height=160,
            usage=spy.TextureUsage.shader_resource,data=data)
        r.bloom_extract.dispatch(thread_count=[64,40,1],vars=dict(hdrSource=source,destination=r.bloom_buffers[0],
            size=spy.uint2(256,160),bloomSize=spy.uint2(64,40),exposure=1.,threshold=4.))
        pixels=r.bloom_buffers[0].to_numpy()[:,:,:3]
        assert np.isfinite(pixels).all()
        peaks.append(float(pixels.max()))
    assert peaks[0]==0 and 0<peaks[1]<peaks[2]<peaks[3]<=16.01


def test_shadow_filter_is_continuous_between_texels(flat_renderer):
    r=flat_renderer;d=r.device
    depth=np.zeros((64,64,2),np.float32);depth[:,:,0]=.75;depth[:,:32,0]=.25
    texture=d.create_texture(format=spy.Format.rg32_float,width=64,height=64,
        usage=spy.TextureUsage.shader_resource,data=depth)
    result=d.create_buffer(size=256*16,struct_size=16,usage=spy.BufferUsage.unordered_access)
    kernel=d.create_compute_kernel(d.load_program(str(ROOT/'tests/shadow_filter.slang'),['filterTest']))
    kernel.dispatch(thread_count=[256,1,1],vars=dict(probeDepth=texture,filterResult=result))
    visibility=result.to_numpy().view(np.float32).reshape(-1,4)[:,0]
    assert visibility[0]<.01 and visibility[-1]>.99
    assert np.diff(visibility).min()>-1e-5
    assert np.diff(visibility).max()<.015  # old sparse PCF jumped by 0.2


@pytest.mark.parametrize('slope',[-12.,12.])
def test_grazing_receiver_uses_its_exact_plane(flat_renderer,slope):
    d=flat_renderer.device
    depth=np.zeros((128,128,2),np.float32)
    depth[:,:,0]=.5+slope*((np.arange(128)+.5)/128*2-1)
    texture=d.create_texture(format=spy.Format.rg32_float,width=128,height=128,
        usage=spy.TextureUsage.shader_resource,data=depth)
    result=d.create_buffer(size=256*16,struct_size=16,usage=spy.BufferUsage.unordered_access)
    kernel=d.create_compute_kernel(d.load_program(str(ROOT/'tests/shadow_filter.slang'),['grazingPlaneTest']))
    kernel.dispatch(thread_count=[256,1,1],vars=dict(probeDepth=texture,filterResult=result,probeSlope=slope))
    coverage=result.to_numpy().view(np.float32).reshape(-1,4)[:,:2]
    assert np.isfinite(coverage).all()
    assert coverage.min()>.999  # clamping the slope to +/-8 invented shadows


def test_printed_sphere_boundaries_have_no_dark_spikes():
    r=Renderer(showroom(),1280,800,shadow_size=2048)
    s=Style(penumbra_strength=0)
    r.render([4,4,-7],[0,.5,1],s,debug=11)
    p=r.gbuffers[0].to_numpy();encoded=r.gbuffers[4].to_numpy()[:,:,:2]
    n=np.concatenate([encoded,(1-abs(encoded).sum(axis=2))[:,:,None]],axis=2)
    folded=(1-abs(encoded[:,:,::-1]))*np.where(encoded>=0,1,-1)
    n[:,:,:2]=np.where((n[:,:,2]<0)[:,:,None],folded,encoded)
    n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-8)
    from inkforge.environment import lighting_state
    nl=n@lighting_state(r.scene,s)['direction']
    # Keep the lit side immediately beside the printed boundary, where the old
    # weaker test (N.L > .2) missed tiny triangular shadow spikes.
    lit=(p[:,:,3]!=0)&(p[:,:,1]>.2)&(nl>.16)&(nl<.3)
    assert lit.sum()>1000
    assert np.min(r.output.to_numpy()[:,:,0][lit])>.98


@pytest.mark.parametrize('outlined',[False,True])
def test_taa_reduces_silhouette_error_and_rejects_camera_cuts(outlined):
    from tools.check_aa import measure
    report,images=measure(outlined)
    assert all(np.isfinite(im).all() for im in images)
    assert report['edge_pixels']>500
    assert report['taa_edge_mae']<report['raw_edge_mae']*.8
    # RGBA16 history adds at most half-float rounding on an invalid history frame.
    assert report['camera_cut_error']<.001
    assert report['style_reset_error']<.001


def test_sphere_shadow_terminator_is_clean():
    from inkforge.procedural import sphere
    from inkforge.math3d import normalize
    v,i=sphere(np.zeros(3),radius=1.)
    scene=Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Sphere terminator regression')
    r=Renderer(scene,320,200,shadow_size=2048);s=Style()
    r.render([2,1,-4],[0,0,0],s,debug=4)
    pos=r.gbuffers[0].to_numpy();geometry=pos[:,:,3]!=0
    from inkforge.environment import lighting_state
    nl=np.sum(normalize(pos[:,:,:3])*lighting_state(scene,s)['direction'],axis=2)
    shadow=r.output.to_numpy()[:,:,0]>.735357 # 0.5 linear, encoded to sRGB
    # Exclude the narrow transition; facet-shaped lit fingers must not cross it.
    dark=geometry&(nl<.08);lit=geometry&(nl>.2)
    assert dark.sum()>500 and lit.sum()>500
    assert np.mean(shadow[dark])<.01
    assert np.mean(shadow[lit])>.98


def test_resize_reuses_scene_resources_and_resets_history(flat_renderer):
    r=flat_renderer;meshes=r.meshes;shadows=r.shadows;device=r.device
    original=(r.width,r.height)
    r.render([0,2,5],[0,0,0],Style())
    r.resize(317,193)
    assert r.meshes is meshes and r.shadows is shadows and r.device is device
    assert r.taa_frame==0
    r.render([0,2,5],[0,0,0],Style())
    assert r.output.to_numpy().shape==(193,317,4)
    assert np.isfinite(r.output.to_numpy()).all()
    r.resize(*original)


def test_continuous_time_keeps_taa_but_large_jump_resets(flat_renderer):
    r=flat_renderer;s=Style()
    for hour in (10.,10.001,10.002):
        s.time_of_day=hour;r.render([0,2,5],[0,0,0],s)
    assert r.taa_frame>=3
    s.time_of_day=17.;r.render([0,2,5],[0,0,0],s)
    assert r.taa_frame==1


@pytest.mark.parametrize('backdrop',[True,False])
def test_distant_subpixel_ink_preserves_coverage_without_trails(backdrop):
    from tools.distant_ink_review import measure
    report,images=measure(backdrop)
    assert all(np.isfinite(image).all() for image in images.values())
    # The old rejection/clipping flashed these rails between ink and paper;
    # its stationary 95th-percentile temporal deviation was 0.064–0.100.
    assert report['stationary']['temporal_std_p95']<.02
    assert report['stationary']['reference_mae']<.012
    assert report['pan']['reference_mae']<.025
    # Stable coverage must not be bought by smearing rails onto clear paper.
    assert report['pan']['clear_background_error_p99']<.03
    assert report['fast_pan']['clear_background_error_p99']<.03
    assert report['cut_history_samples']==1


def test_stationary_jitter_cycle_is_stable_but_lighting_stays_live():
    r=Renderer(showroom(),320,200,shadow_size=256)
    s=Style(bloom=0)
    frames=[]
    for frame in range(24):
        r.render([4,4,-7],[0,.5,1],s)
        if frame>=8:frames.append(r.output.to_numpy()[:,:,:3].copy())
    # Every output now integrates the same eight raster phases, instead of
    # fluctuating as a recursive history filter rejects/clips different edges.
    assert np.max(np.std(frames,axis=0))<2e-5
    before=frames[-1]
    for frame in range(8):
        s.time_of_day+=.02
        r.render([4,4,-7],[0,.5,1],s,time_seconds=frame*2)
    assert r.stationary_frame>=8
    assert np.mean(abs(r.output.to_numpy()[:,:,:3]-before))>1e-4
    r.render([4.01,4,-7],[.01,.5,1],s)
    assert r.stationary_frame==0
    for _ in range(8):r.render([4.01,4,-7],[.01,.5,1],s)
    assert r.stationary_frame==8
    s.outline+=.1;r.render([4.01,4,-7],[.01,.5,1],s)
    assert r.stationary_frame==1


def test_tiny_print_disks_preserve_coverage(flat_renderer):
    d=flat_renderer.device
    kernel=d.create_compute_kernel(d.load_program(str(ROOT/'tests/penumbra_probe.slang'),['diskAreaTest']))
    target=d.create_texture(format=spy.Format.r32_float,width=256,height=256,
        usage=spy.TextureUsage.unordered_access|spy.TextureUsage.shader_resource)
    for radius in (.03,.1,.3):
        kernel.dispatch(thread_count=[256,256,1],vars=dict(result=target,diskRadius=radius))
        integrated=float(target.to_numpy().sum())*(1.5/256)**2
        assert integrated==pytest.approx(np.pi*radius**2,rel=.005)
