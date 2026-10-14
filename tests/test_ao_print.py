"""Check rendered hatch data, not only the source texture assets."""
import numpy as np
import slangpy as spy
from inkforge.renderer import Renderer,Style
from inkforge.procedural import plane
from inkforge.scene import Scene,Primitive,Material

class Skip:
    def dispatch(self,**kwargs):pass

def test_production_hatches_are_sparse_directional_strokes():
    v,i=plane(2)
    r=Renderer(Scene([Primitive(v,i,0)],[Material(metallic=0)],[],'Hatch binding probe'),321,201,shadow_size=128)
    # Hold occlusion constant to isolate printing from screen-space geometry.
    r.gtao=r.denoise=r.ao_upsample=Skip()
    enc=r.device.create_command_encoder();enc.clear_texture_float(r.ao,clear_value=[.3,0,0,0]);r.device.submit_command_buffer(enc.finish())
    results=[]
    for direction in (0,2):
        r.render([0,.35,0],[0,0,.001],Style(taa=False,hatch_direction=direction),debug=6)
        positions=r.gbuffers[0].to_numpy();pixels=r.output.to_numpy()[:,:,0]
        mask=(positions[:,:,3]!=0)&(abs(positions[:,:,0])<1)&(abs(positions[:,:,2])<.5)
        linear=np.where(pixels<=.04045,pixels/12.92,((pixels+.055)/1.055)**2.4)
        result=linear[mask];results.append(result)
        assert .01<result.mean()<.2, (result.mean(),result.max())
        assert result.std()>.025
        assert np.mean(result>.9)<.01
    assert abs(results[0]-results[1]).mean()>.015


def test_resolved_hatch_sites_do_not_change_with_camera_footprint():
    from pathlib import Path
    v,i=plane(1);r=Renderer(Scene([Primitive(v,i,0)],[Material()],[],'Hatch phase probe'),32,32,shadow_size=64)
    # Compile the actual production functions in a small resource-complete probe.
    text=(Path(__file__).resolve().parents[1]/'inkforge/shaders/lighting.slang').read_text()
    functions=text[text.index('float4 sampleHatch('):text.index('[shader("compute")]')]
    source='''Texture2D<float4> hatchTextures[4]; SamplerState hatchSampler;
int hatchDirection; float4 effects; float hatchDensity;
RWStructuredBuffer<float4> result;
'''+functions+'''
[shader("compute")][numthreads(64,1,1)]void probe(uint3 t:SV_DispatchThreadID){
    if(t.x>=256)return;
    float2 uv=float2(t.x*.00713,t.x*.00517);
    float4 value;
    for(int i=0;i<4;i++){
        float footprint=.00002*exp2(float(i));
        value[i]=crosshatch(uv,float2(footprint,0),float2(0,footprint),.85);
    }
    result[t.x]=value;
}'''
    module=r.device.load_module_from_source('hatch_phase_probe',source)
    kernel=r.device.create_compute_kernel(r.device.link_program([module],[module.entry_point('probe')]))
    output=r.device.create_buffer(size=256*16,usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[256,1,1],vars=dict(hatchTextures=r.hatch_textures,hatchSampler=r.hatch_sampler,
        hatchDirection=0,effects=spy.float4(0,0,0,32),hatchDensity=2.,result=output))
    values=output.to_numpy().view(np.float32).reshape(-1,4)
    assert values[:,0].std()>.05
    assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
    np.testing.assert_allclose(values,np.repeat(values[:,:1],4,axis=1),atol=1e-5)


def test_default_bloom_print_remains_visible_on_bright_receivers():
    # Production 3px sites / .04 strength, not the oversized dots used by the
    # density test. Tiny highlights must survive a bright mapped background.
    v,i=plane(1);r=Renderer(Scene([Primitive(v,i,0)],[Material()],[],'Bloom contrast probe'),256,160,shadow_size=64)
    d=r.device
    source=d.create_texture(format=spy.Format.rgba32_float,width=64,height=40,
        usage=spy.TextureUsage.shader_resource,data=np.full((40,64,4),.8,np.float32))
    for level in (.02,.35,.75):
        data=np.full((160,256,4),level,np.float32);data[:,:,3]=1
        base=d.create_texture(format=spy.Format.rgba32_float,width=256,height=160,
            usage=spy.TextureUsage.shader_resource,data=data)
        params=dict(baseSource=base,source=source,linearSampler=r.bloom_sampler,finalImage=r.output,
            size=spy.uint2(256,160),bloomSize=spy.uint2(64,40),dotSize=3.)
        r.bloom_composite.dispatch(thread_count=[256,160,1],vars=dict(**params,strength=0.))
        plain=r.output.to_numpy()[:,:,:3].copy()
        r.bloom_composite.dispatch(thread_count=[256,160,1],vars=dict(**params,strength=.04))
        printed=r.output.to_numpy()[:,:,:3]
        delta=printed-plain
        assert np.isfinite(printed).all() and printed.max()<=1
        assert delta.min()>-1e-6
        assert np.mean(delta[:,:,0]>1/255)>.035
        assert np.mean(delta[:,:,0]<1e-5)>.2  # gaps, not a bloom wash
        # Display-linear addition is bounded by the remaining highlight room.
        linear=np.where(printed<=.04045,printed/12.92,((printed+.055)/1.055)**2.4)
        assert np.max(linear-level)<=(1-level)*.201
