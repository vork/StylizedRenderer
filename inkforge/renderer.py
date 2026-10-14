"""Native raster G-buffer -> GTAO -> stylized lighting -> GT7 SDR."""
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import json
import platform
import time
import numpy as np
from PIL import Image
import slangpy as spy
from .math3d import normalize, look_at, perspective, orthographic
from .textures import mip_chain
from .shadows import cascades
from .environment import lighting_state,pack_lights


@dataclass
class Style:
    amount: float = 1.
    outline: float = 1.
    outline_wobble: float = .3
    taa: bool = True
    penumbra_strength: float = .55
    shadow_fill: float = .6
    lighting_mode: str = 'auto'
    time_of_day: float = 10.
    halftone: float = 0.
    specular_size: float = .10
    exposure: float = 1.6
    ao: float = 1.3
    ambient: float = 1.5
    sun: float = 4.
    rim: float = .22
    saturation: float = 1.25
    fog: float = .002
    dot_frequency: float = 32.
    ao_radius: float = 1.2
    hatching: float = .75
    hatch_density: float = 2.
    cloud_shadows: float = .4
    bloom: float = .04
    hatch_direction: int = 0
    bloom_threshold: float = 3.
    bloom_dot_size: float = 3.
    palette: float = .55
    texture_bias: float = 1.2
    shadow_distance: float = 100.
    shadow_tint: tuple = (.25, .22, .48)
    sky: tuple = (.32, .65, .82)
    sun_color: tuple = (1., .86, .66)
    sun_direction: tuple = (-.5, .8, .35)


def preset(name):
    s = Style()
    if name == 'pbr':
        s.amount = 0.; s.saturation = 1.; s.ao = 1.; s.texture_bias=0.; s.palette=0.; s.hatching=0.;s.bloom=0.
    elif name == 'soft':
        s.amount = .65; s.outline = .65; s.hatching = .18; s.saturation = 1.05; s.specular_size = .04
    elif name == 'graphic':
        s.outline = 2.; s.saturation = 1.4; s.specular_size = .16; s.palette=.9; s.hatching=.42; s.texture_bias=2.
    elif name != 'rush':
        raise ValueError(name)
    return s


def default_camera(scene):
    low, high = scene.bounds
    center = (low+high)*.5
    extent = high-low
    radius = max(extent)*.7
    return center+np.array([.8,.45,1.])*radius, center


class Renderer:
    def __init__(self, scene, width=1280, height=720, backend='auto', max_texture=1024, shadow_size=2048,device=None):
        self.scene, self.width, self.height = scene, width, height
        self.bounds=scene.bounds
        self.active_style=Style()
        self.temporal_coverage=True # diagnostic A/B switch; on in the viewer
        self.stationary_resolve=True # exact jitter-cycle average while the camera rests
        self.motion_jitter_scale=.25 # avoid a full pixel's sampling crawl during camera motion
        if backend == 'auto':
            backend = 'metal' if platform.system() == 'Darwin' else 'd3d12' if platform.system() == 'Windows' else 'vulkan'
        self.backend = backend
        self.device = device or spy.Device(type=getattr(spy.DeviceType, backend), compiler_options={
            'include_paths': [str(Path(__file__).parent/'shaders')],
            'matrix_layout': spy.SlangMatrixLayout.row_major,
        })
        d = self.device
        self.max_texture=max_texture;self.shadow_size=shadow_size
        self.scene_lights=d.create_buffer(data=pack_lights(scene.lights),usage=spy.BufferUsage.shader_resource,struct_size=64)
        print(f'GPU: {d.info.adapter_name} / {backend}', flush=True)
        self.layout = d.create_input_layout(input_elements=[
            {'semantic_name': name, 'semantic_index': index, 'format': fmt, 'offset': offset}
            for name,index,fmt,offset in [('POSITION',0,spy.Format.rgb32_float,0),
                ('NORMAL',0,spy.Format.rgb32_float,12),('TEXCOORD',0,spy.Format.rg32_float,24),
                ('TEXCOORD',1,spy.Format.rg32_float,32),('TANGENT',0,spy.Format.rgba32_float,40),
                ('COLOR',0,spy.Format.rgba32_float,56),('TEXCOORD',2,spy.Format.rg32_float,72)]], vertex_streams=[{'stride':80}])
        depth = {'format':spy.Format.d32_float,'depth_test_enable':True,'depth_write_enable':True,'depth_func':spy.ComparisonFunc.less}
        self.gprogram = d.load_program('geometry.slang',['vertexMain','fragmentSingle' if backend=='metal' else 'fragmentMain'])
        self.storage_geometry=backend=='metal'
        self.storage_program=d.load_program('geometry.slang',['vertexMain','fragmentStorage']) if backend=='metal' else None
        self.depth_program=d.load_program('geometry.slang',['vertexMain','depthMain']) if backend=='metal' else None
        self.sprogram = d.load_program('geometry.slang',['vertexMain','shadowMain'])
        formats = [spy.Format.rgba32_float] + [spy.Format.rgba16_float]*3 + [spy.Format.rgba32_float]
        self.pipelines = {}
        for shadow in (False,True):
            for double in (False,True):
                self.pipelines[shadow,double] = d.create_render_pipeline(
                    program=self.sprogram if shadow else self.gprogram, input_layout=self.layout,
                    targets=[{'format':f} for f in ([spy.Format.rg32_float] if shadow else formats[:1] if backend=='metal' else formats)],
                    depth_stencil=depth, rasterizer={'cull_mode':spy.CullMode.none if double else spy.CullMode.back,
                                                    'front_face':spy.FrontFaceMode.counter_clockwise})
                if not shadow and backend=='metal':
                    self.pipelines['half',double]=d.create_render_pipeline(program=self.gprogram,input_layout=self.layout,
                        targets=[{'format':spy.Format.rgba16_float}],depth_stencil=depth,
                        rasterizer={'cull_mode':spy.CullMode.none if double else spy.CullMode.back,'front_face':spy.FrontFaceMode.counter_clockwise})
        self.lighting = d.create_compute_kernel(d.load_program('lighting.slang',['lightingMain']))
        self.shadow_mask=d.create_compute_kernel(d.load_program('shadow_mask.slang',['shadowMaskMain']))
        if backend=='metal':
            for double in (False,True):
                for stage,program in [('depth',self.depth_program),('storage',self.storage_program)]:
                    self.pipelines[stage,double]=d.create_render_pipeline(program=program,input_layout=self.layout,
                        targets=[{'format':spy.Format.rgba32_float}],
                        depth_stencil={**depth,'depth_write_enable':stage=='depth',
                            'depth_func':spy.ComparisonFunc.less if stage=='depth' else spy.ComparisonFunc.equal},
                        rasterizer={'cull_mode':spy.CullMode.none if double else spy.CullMode.back,
                                    'front_face':spy.FrontFaceMode.counter_clockwise})
        self.gtao = d.create_compute_kernel(d.load_program('gtao.slang',['gtaoMain']))
        self.denoise = d.create_compute_kernel(d.load_program('gtao.slang',['denoiseMain']))
        self.ao_upsample=d.create_compute_kernel(d.load_program('gtao.slang',['upsampleMain']))
        self.bloom_extract=d.create_compute_kernel(d.load_program('bloom.slang',['extractMain']))
        self.bloom_blur=d.create_compute_kernel(d.load_program('bloom.slang',['blurMain']))
        self.bloom_composite=d.create_compute_kernel(d.load_program('bloom.slang',['compositeMain']))
        self.tonemap=d.create_compute_kernel(d.load_program('bloom.slang',['tonemapMain']))
        self.taa_kernel=d.create_compute_kernel(d.load_program('taa.slang',['taaMain']))
        self.edge_aa=d.create_compute_kernel(d.load_program('edge_aa.slang',['edgeAAMain']))
        self.shadows = [self.texture(spy.Format.rg32_float,shadow_size,shadow_size,True) for _ in range(4)]
        self.shadow_depths = [d.create_texture(format=spy.Format.d32_float,width=shadow_size,height=shadow_size,usage=spy.TextureUsage.depth_stencil) for _ in range(4)]
        self.cascade_index=0
        self.resize(width,height,initial=True)
        self.bloom_sampler=d.create_sampler(min_filter=spy.TextureFilteringMode.linear,mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,address_v=spy.TextureAddressingMode.clamp_to_edge)
        self.hatch_sampler=d.create_sampler(min_filter=spy.TextureFilteringMode.linear,mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,address_u=spy.TextureAddressingMode.wrap,address_v=spy.TextureAddressingMode.wrap)
        self.hatch_textures=[]
        for index in range(4):
            im=Image.open(Path(__file__).parent/'resources'/f'hatching-{index}.png')
            # Independent data channels: never premultiply RGB by the fourth direction.
            levels=[]
            while True:
                levels.append(np.asarray(im).copy())
                if im.width==1 and im.height==1:break
                im=Image.merge('RGBA',[c.resize((max(1,im.width//2),max(1,im.height//2)),Image.Resampling.BOX) for c in im.split()])
            texture=d.create_texture(format=spy.Format.rgba8_unorm,width=levels[0].shape[1],height=levels[0].shape[0],
                mip_count=len(levels),usage=spy.TextureUsage.shader_resource)
            enc=d.create_command_encoder();enc.upload_texture_data(texture,levels);d.submit_command_buffer(enc.finish())
            self.hatch_textures.append(texture)
        self.textures, self.materials, self.samplers = {}, [], {}
        defaults = [(255,255,255,255),(255,255,255,255),(128,128,255,255),(255,255,255,255),(255,255,255,255)]
        for mat in scene.materials:
            tex, samplers = [], []
            for slot, image_id in enumerate(mat.textures):
                srgb=slot in (0,4)
                key = (image_id,srgb) if image_id is not None else ('default',slot)
                if key not in self.textures:
                    im = Image.open(io.BytesIO(scene.images[image_id])).convert('RGBA') if image_id is not None else Image.new('RGBA',(1,1),defaults[slot])
                    levels=mip_chain(im,max_texture,srgb)
                    texture = d.create_texture(format=spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm,
                                               width=levels[0].shape[1],height=levels[0].shape[0],
                                               mip_count=len(levels),usage=spy.TextureUsage.shader_resource)
                    enc=d.create_command_encoder(); enc.upload_texture_data(texture,levels); d.submit_command_buffer(enc.finish())
                    self.textures[key]=texture
                tex.append(self.textures[key])
                wrap=mat.wraps[slot]
                if wrap not in self.samplers:
                    modes={10497:spy.TextureAddressingMode.wrap,33071:spy.TextureAddressingMode.clamp_to_edge,33648:spy.TextureAddressingMode.mirror_repeat}
                    self.samplers[wrap]=d.create_sampler(min_filter=spy.TextureFilteringMode.linear,mag_filter=spy.TextureFilteringMode.linear,
                        mip_filter=spy.TextureFilteringMode.linear,address_u=modes[wrap[0]],address_v=modes[wrap[1]])
                samplers.append(self.samplers[wrap])
            rows=[]
            for transform, coord in zip(mat.transforms,mat.texcoords):
                rows.extend([transform[:3]+[float(coord)],transform[3:]+[float(coord)]])
            self.materials.append((tex,samplers,np.asarray(rows,np.float32)))
        self.meshes=[]
        for p in scene.primitives:
            if not len(p.indices): continue
            triangles=p.vertices[p.indices.reshape(-1,3),:3].astype(np.float64)
            faces=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            lengths=np.linalg.norm(faces,axis=1)
            faces[lengths<1e-15]=[0,1,0]
            faces/=np.maximum(np.linalg.norm(faces,axis=1,keepdims=True),1e-15)
            packed_faces=np.pad(faces.astype(np.float32),((0,0),(0,1)))
            self.meshes.append((d.create_buffer(usage=spy.BufferUsage.vertex_buffer,data=np.ascontiguousarray(p.vertices)),
                                d.create_buffer(usage=spy.BufferUsage.index_buffer,data=np.ascontiguousarray(p.indices)),
                                len(p.indices),p.material,d.create_buffer(usage=spy.BufferUsage.shader_resource,data=packed_faces,struct_size=16)))
        d.wait()
        self.shadow_key=None
        self.binding_cache={}
        print(f'Loaded {scene.triangle_count:,} triangles, {len(self.meshes)} material batches, {len(self.textures)} textures', flush=True)

    def texture(self,fmt,w,h,render=False):
        usage=spy.TextureUsage.shader_resource|spy.TextureUsage.unordered_access|(spy.TextureUsage.render_target if render else spy.TextureUsage(0))
        return self.device.create_texture(format=fmt,width=w,height=h,usage=usage)

    def resize(self,width,height,initial=False):
        """Replace only viewport resources; keep uploaded scenes and shadow maps."""
        width,height=int(width),int(height)
        if width<1 or height<1:return
        if not initial and (width,height)==(self.width,self.height):return
        if not initial:self.device.wait()
        formats=[spy.Format.rgba32_float]+[spy.Format.rgba16_float]*3+[spy.Format.rgba32_float]
        self.gbuffers=[self.texture(f,width,height,True) for f in formats]
        self.depth=self.device.create_texture(format=spy.Format.d32_float,width=width,height=height,usage=spy.TextureUsage.depth_stencil)
        self.ao_raw=self.texture(spy.Format.r32_float,(width+1)//2,(height+1)//2)
        self.ao_filtered=self.texture(spy.Format.r32_float,(width+1)//2,(height+1)//2)
        self.ao=self.texture(spy.Format.r32_float,width,height)
        self.cast_visibility=self.texture(spy.Format.r32_float,width,height)
        self.color=self.texture(spy.Format.rgba32_float,width,height);self.output=self.color
        self.mapped=self.texture(spy.Format.rgba16_float,width,height)
        self.antialiased=self.texture(spy.Format.rgba16_float,width,height)
        self.ink_mask=self.texture(spy.Format.r16_float,width,height)
        self.history_color=[self.texture(spy.Format.rgba16_float,width,height) for _ in range(2)]
        self.history_world=[self.texture(spy.Format.rgba32_float,width,height) for _ in range(2)]
        self.stationary_samples=None # Allocate only when temporal AA is used.
        self.hdr=self.texture(spy.Format.rgba16_float,width,height)
        self.bloom_buffers=[self.texture(spy.Format.rgba16_float,(width+3)//4,(height+3)//4) for _ in range(2)]
        self.width,self.height=width,height
        self.taa_frame=0;self.taa_key=None;self.previous_vp=None;self.previous_eye=None;self.previous_forward=None
        self.previous_hour=None
        self.stationary_frame=0

    def raster(self, encoder, vp, shadow=False, target_index=None):
        # slang-rhi currently mistakes MRT attachment count for array-layer count
        # on Metal. Use single-target passes there; Vulkan/D3D12 retain MRT.
        if not shadow and self.backend=='metal' and target_index is None:
            if self.storage_geometry:
                for target in self.gbuffers[1:]:encoder.clear_texture_float(target,clear_value=[0,0,0,0])
                self.raster(encoder,vp,False,'depth')
                self.raster(encoder,vp,False,'storage')
            else:
                for i in range(5): self.raster(encoder,vp,False,i)
            return
        storage=target_index=='storage';prepass=target_index=='depth'
        targets=[self.shadows[self.cascade_index]] if shadow else [self.gbuffers[0]] if storage or prepass else [self.gbuffers[target_index]] if target_index is not None else self.gbuffers
        depth=self.shadow_depths[self.cascade_index] if shadow else self.depth
        w,h=targets[0].width,targets[0].height
        desc={'color_attachments':[{'view':t.create_view({}),'load_op':spy.LoadOp.load if storage else spy.LoadOp.clear,
                'store_op':spy.StoreOp.store,'clear_value':[1,0,0,0] if shadow else [0,0,0,0]} for t in targets],
              'depth_stencil_attachment':{'view':depth.create_view({}),'depth_load_op':spy.LoadOp.load if storage else spy.LoadOp.clear,'depth_store_op':spy.StoreOp.store,'depth_clear_value':1.}}
        with encoder.begin_render_pass(desc) as rp:
            for batch,(vb,ib,count,mid,faces) in enumerate(self.meshes):
                mat=self.scene.materials[mid]
                # Separate objects per pass prevent later cascade/target updates
                # from modifying bindings already recorded into this encoder.
                binding_key=(shadow,self.cascade_index if shadow else target_index,batch)
                style_key=(self.active_style.texture_bias,self.active_style.amount)
                cached=self.binding_cache.get(binding_key)
                if cached is None:
                    root=self.device.create_root_shader_object(self.sprogram if shadow else self.storage_program if storage else self.depth_program if prepass else self.gprogram)
                    c=spy.ShaderCursor(root)
                    c.baseFactor=spy.float4(mat.base)
                    c.alphaParams=spy.float2({'OPAQUE':0,'MASK':1,'BLEND':2}[mat.alpha_mode],mat.cutoff)
                    textures,samplers,rows=self.materials[mid]
                    c.baseTexture=textures[0]
                    c.triangleNormals=faces
                    for i in range(5): c.texSamplers[i]=samplers[i]
                    for i,row in enumerate(rows): c.uvRows[i]=spy.float4(row)
                    if not shadow:
                        if target_index is not None and not (storage or prepass):c.targetIndex=target_index
                        c.emission=spy.float4(*mat.emissive,float(mat.unlit))
                        c.mrTexture=textures[1];c.normalTexture=textures[2];c.aoTexture=textures[3];c.emissiveTexture=textures[4]
                else:
                    root,c,_=cached
                if cached is None or cached[2]!=style_key:
                    c.textureBias=self.active_style.texture_bias*self.active_style.amount
                    if not shadow:
                        c.materialParams=spy.float4(mat.metallic,mat.roughness,mat.normal_scale*(1-self.active_style.amount*.55),mat.occlusion)
                    self.binding_cache[binding_key]=(root,c,style_key)
                c.viewProjection=spy.float4x4(vp)
                if storage:
                    c.storedNormal=self.gbuffers[1];c.storedBase=self.gbuffers[2]
                    c.storedEmission=self.gbuffers[3];c.storedGeometryNormal=self.gbuffers[4]
                pipeline=target_index if storage or prepass else 'half' if target_index in (1,2,3) else shadow
                rp.bind_pipeline(self.pipelines[pipeline,mat.double_sided],root)
                rp.set_render_state({'vertex_buffers':[vb],'index_buffer':ib,'index_format':spy.IndexFormat.uint32,
                    'viewports':[spy.Viewport.from_size(w,h)],'scissor_rects':[spy.ScissorRect.from_size(w,h)]})
                rp.draw_indexed({'vertex_count':count})

    def render(self, eye, target, style=None, fov=55., debug=0, encoder=None,time_seconds=0.):
        style=style or Style()
        self.active_style=style
        low,high=self.bounds
        extent=float(np.linalg.norm(high-low)); center=(high+low)/2
        view=look_at(eye,target)
        proj=perspective(fov,self.width/self.height,max(.005,extent*.0001),max(extent*8,100))
        use_taa=style.taa and debug==0
        forward=normalize(np.asarray(target)-eye)
        camera_cut=self.previous_eye is not None and (np.linalg.norm(np.asarray(eye)-self.previous_eye)>max(1.,extent*.05) or np.dot(forward,self.previous_forward)<.85)
        style_key=asdict(style);style_key.pop('time_of_day')
        hour_delta=0 if self.previous_hour is None else abs((style.time_of_day-self.previous_hour+12)%24-12)
        camera_cut=camera_cut or hour_delta>.25
        taa_key=(json.dumps(style_key,sort_keys=True),fov,self.temporal_coverage,self.stationary_resolve,self.motion_jitter_scale) if use_taa else None
        if not use_taa or taa_key!=self.taa_key or camera_cut:
            self.taa_frame=0;self.previous_vp=None
        self.taa_key=taa_key
        unjittered_vp=proj@view
        same_camera=self.previous_vp is not None and np.array_equal(unjittered_vp,self.previous_vp)
        camera_moving=self.previous_vp is not None and not same_camera
        self.stationary_frame=self.stationary_frame+1 if use_taa and same_camera else 1 if use_taa and not camera_moving else 0
        jitter=np.zeros(2,np.float32)
        if use_taa:
            def halton(index,base):
                result=0.;weight=1.
                while index:
                    weight/=base;result+=weight*(index%base);index//=base
                return result
            sample=self.taa_frame%8+1
            jitter=np.array([halton(sample,2)-.5,halton(sample,3)-.5],np.float32)
            if camera_moving:jitter*=self.motion_jitter_scale
            proj[0,2]-=2*jitter[0]/self.width
            proj[1,2]+=2*jitter[1]/self.height
        vp=proj@view
        environment=lighting_state(self.scene,style)
        light=normalize(environment['direction'])
        light_matrices,splits=cascades(eye,target,fov,self.width/self.height,max(.005,extent*.0001),
            min(style.shadow_distance,max(extent,20)),light,self.bounds,self.shadows[0].width)
        own=encoder is None
        enc=encoder or self.device.create_command_encoder()
        key=(light_matrices.tobytes(),style.texture_bias,style.amount)
        if key!=self.shadow_key:
            for i in range(4):
                self.cascade_index=i;self.raster(enc,light_matrices[i],True)
            self.shadow_key=key
        self.raster(enc,vp)
        self.shadow_mask.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,vars=dict(
            positions=self.gbuffers[0],outputShadow=self.cast_visibility,shadowMaps=self.shadows,
            size=spy.uint2(self.width,self.height),shadowBias=.000015,
            lightMatrices=[spy.float4x4(m) for m in light_matrices],
            lightNormalMatrices=[spy.float3x3(np.linalg.inv(m[:3,:3]).T) for m in light_matrices],
            shadowSplits=spy.float4(splits),cameraView=spy.float4x4(view)))
        ao_size=spy.uint2(self.ao_raw.width,self.ao_raw.height)
        ao_params=dict(positions=self.gbuffers[0],normals=self.gbuffers[4],size=ao_size,
            fullSize=spy.uint2(self.width,self.height),radius=style.ao_radius)
        self.gtao.dispatch(thread_count=[*ao_size,1],command_encoder=enc,vars=dict(
            **ao_params,outputAO=self.ao_raw,view=spy.float4x4(view),focalPixels=float(proj[1,1]*self.height*.5)))
        self.denoise.dispatch(thread_count=[*ao_size,1],command_encoder=enc,vars=dict(
            **ao_params,sourceAO=self.ao_raw,outputAO=self.ao_filtered))
        self.ao_upsample.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,vars=dict(
            **ao_params,sourceAO=self.ao_filtered,outputAO=self.ao))
        self.lighting.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,vars=dict(
            positions=self.gbuffers[0],normals=self.gbuffers[1],bases=self.gbuffers[2],emissions=self.gbuffers[3],
            geometryNormals=self.gbuffers[4],ao=self.ao,castVisibilityMap=self.cast_visibility,
            worldPixelScale=float(2/(self.height*proj[1,1])),outputImage=self.color,hdrImage=self.hdr,inkMask=self.ink_mask,
            size=spy.uint2(self.width,self.height),eye=spy.float3(eye),sunDirection=spy.float3(light),sunColor=spy.float3(environment['color']),
            sceneLights=self.scene_lights,sceneLightCount=len(self.scene.lights) if environment['keep'] else 0,skipLightIndex=environment['skip'],
            skySunDirection=spy.float3(environment['sun_direction']),skyDaylight=environment['daylight'],
            skyTime=float(time_seconds),skyEye=spy.float3(eye),skyPixelAngle=float(1/(self.height*proj[1,1])),proceduralSky=environment['procedural'],
            cloudShadowStrength=style.cloud_shadows if environment['cloud_key'] else 0.,
            shadowSplits=spy.float4(splits),cameraView=spy.float4x4(view),inverseVP=spy.float4x4(np.linalg.inv(vp)),
            style=spy.float4(style.amount,style.outline,style.halftone,style.specular_size),
            settings=spy.float4(style.exposure,style.ao,style.ambient,environment['power']),
            effects=spy.float4(style.rim,style.saturation,style.fog,style.dot_frequency),
            comic=spy.float2(style.hatching,style.palette),
            outlineWobble=style.outline_wobble,hatchDensity=style.hatch_density,
            penumbraStrength=style.penumbra_strength,shadowFillLevel=1-(1-style.shadow_fill)*environment['shadow_presence'],
            hatchTextures=self.hatch_textures,hatchSampler=self.hatch_sampler,hatchDirection=style.hatch_direction,
            shadowTint=spy.float3(environment['tint']),skyColor=spy.float3(environment['sky']),debugView=debug))
        # Resolve scene edges in display-linear light before the screen print.
        # Bloom dots no longer force low history weights on nearby outlines.
        resolved_base=self.mapped
        if debug==0:
            self.tonemap.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,
                vars=dict(size=spy.uint2(self.width,self.height),hdrSource=self.hdr,finalImage=self.mapped,exposure=float(2**style.exposure)))
        if use_taa:
            if self.stationary_samples is None:
                self.stationary_samples=self.device.create_texture(type=spy.TextureType.texture_2d_array,
                    format=spy.Format.rgba16_float,width=self.width,height=self.height,array_length=8,
                    usage=spy.TextureUsage.shader_resource|spy.TextureUsage.unordered_access)
            write=self.taa_frame%2;read=1-write
            self.taa_kernel.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,vars=dict(
                currentColor=self.mapped,positions=self.gbuffers[0],historyColor=self.history_color[read],
                historyWorld=self.history_world[read],inkMask=self.ink_mask,resolved=self.history_color[write],
                nextWorld=self.history_world[write],linearSampler=self.bloom_sampler,size=spy.uint2(self.width,self.height),
                previousVP=spy.float4x4(self.previous_vp if self.previous_vp is not None else unjittered_vp),
                currentVP=spy.float4x4(unjittered_vp),
                inverseVP=spy.float4x4(np.linalg.inv(vp)),jitter=spy.float2(jitter),
                eye=spy.float3(eye),worldPixelScale=float(2/(self.height*proj[1,1])),historyValid=self.taa_frame>0,
                historySamples=min(self.taa_frame,64),coverageEnabled=self.temporal_coverage,
                stationarySamples=self.stationary_samples,phaseIndex=self.taa_frame%8,
                stationaryReady=self.stationary_resolve and self.stationary_frame>=8))
            resolved_base=self.history_color[write]
            self.previous_vp=unjittered_vp.copy();self.taa_frame+=1
            # Temporal accumulation alone leaves high-contrast subpixel lines
            # changing coverage as they pan. Reconstruct those spatial edges
            # before adding the deliberately screen-printed bloom dots.
            self.edge_aa.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,vars=dict(
                source=resolved_base,destination=self.antialiased,linearSampler=self.bloom_sampler,
                size=spy.uint2(self.width,self.height)))
            resolved_base=self.antialiased
        if debug==0:
            bloom0,bloom1=self.bloom_buffers
            params=dict(size=spy.uint2(self.width,self.height),bloomSize=spy.uint2(bloom0.width,bloom0.height),
                exposure=float(2**style.exposure),threshold=style.bloom_threshold,strength=style.bloom*style.amount,dotSize=style.bloom_dot_size)
            if style.bloom*style.amount>0:
                self.bloom_extract.dispatch(thread_count=[bloom0.width,bloom0.height,1],command_encoder=enc,
                    vars=dict(**params,hdrSource=self.hdr,destination=bloom0))
                for source,destination,direction in [(bloom0,bloom1,[1,0]),(bloom1,bloom0,[0,1])]:
                    self.bloom_blur.dispatch(thread_count=[bloom0.width,bloom0.height,1],command_encoder=enc,
                        vars=dict(**params,source=source,destination=destination,direction=spy.int2(direction)))
            self.bloom_composite.dispatch(thread_count=[self.width,self.height,1],command_encoder=enc,
                vars=dict(**params,hdrSource=self.hdr,baseSource=resolved_base,source=bloom0,linearSampler=self.bloom_sampler,finalImage=self.color))
        self.output=self.color
        self.previous_eye=np.asarray(eye).copy();self.previous_forward=forward;self.previous_hour=style.time_of_day
        if own: self.device.submit_command_buffer(enc.finish())
        return self.output

    def save(self,path,eye,target,style,fov=55,debug=0,downsample=1,time_seconds=0.):
        start=time.perf_counter()
        for _ in range(8 if style.taa and debug==0 else 1):
            self.render(eye,target,style,fov,debug,time_seconds=time_seconds)
        pixels=self.output.to_numpy()
        if not np.isfinite(pixels).all(): raise RuntimeError('Non-finite GPU output')
        im=Image.fromarray(np.uint8(np.clip(pixels[:,:,:3],0,1)*255+.5))
        if downsample>1: im=im.resize((self.width//downsample,self.height//downsample),Image.Resampling.LANCZOS)
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); im.save(path)
        metadata={'source':self.scene.source,'backend':self.backend,'gpu':self.device.info.adapter_name,
                  'triangles':self.scene.triangle_count,'batches':len(self.meshes),'eye':list(map(float,eye)),
                  'target':list(map(float,target)),'fov':fov,'style':asdict(style),'render_resolution':[self.width,self.height],
                  'output_resolution':list(im.size),'debug':debug,'elapsed_with_readback_s':time.perf_counter()-start}
        metadata['cloud_time_seconds']=time_seconds
        path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        print(f'Saved {path} ({metadata["elapsed_with_readback_s"]:.2f}s incl. readback)',flush=True)
        return im
