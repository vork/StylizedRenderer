import numpy as np
from PIL import Image
from inkforge.scene import make_vertices,load_scene
from inkforge.procedural import showroom,export_usd
from inkforge.textures import mip_chain


def test_reflected_transform_preserves_winding_and_normal():
    m=np.diag([-2.,3.,.5,1.])
    p=np.array([[0,0,0],[1,0,0],[0,1,1]],np.float32)
    v,i=make_vertices(p,[0,1,2],matrix=m)
    geometric=np.cross(v[i[1],:3]-v[i[0],:3],v[i[2],:3]-v[i[0],:3])
    geometric/=np.linalg.norm(geometric)
    np.testing.assert_allclose(v[0,3:6],geometric,atol=1e-6)


def test_usd_roundtrip(tmp_path):
    original=showroom();path=tmp_path/'scene.usda';export_usd(original,path)
    loaded=load_scene(path)
    assert loaded.triangle_count==original.triangle_count
    np.testing.assert_allclose(loaded.bounds,original.bounds,atol=1e-6)
    assert sorted(m.metallic for m in loaded.materials).count(1.)==4
    assert any(abs(m.roughness-.12)<1e-5 for m in loaded.materials)
    assert any(np.ptp(p.vertices[:,6:8],axis=0).min()>.5 for p in loaded.primitives)


def test_hatch_textures_preserve_strokes_as_density_increases():
    from inkforge.hatching import create_textures
    maps=create_textures()  # Validate the production-resolution stroke assets.
    for low,high in zip(maps,maps[1:]):
        assert np.all(high>=low)
        assert high.mean()>low.mean()
    # Distinct directional channels and genuinely sparse ink.
    assert not np.array_equal(maps[-1][:,:,0],maps[-1][:,:,1])
    assert np.mean(maps[-1]/255)<.3


def test_srgb_mips_filter_in_linear_light():
    image=Image.fromarray(np.array([[[0,0,0,255],[255,255,255,255]]],np.uint8))
    encoded=mip_chain(image,2,True)[-1][0,0]
    assert 187<=encoded[0]<=189  # 50% linear light encodes to ~188, not 128.
    assert encoded[3]==255


def test_gltf_embedded_normalized_and_interleaved(tmp_path):
    import json,base64,struct
    # Interleaved positions with a 16-byte stride, normalized uint8 colors.
    data=struct.pack('<12f',0,0,0,99,1,0,0,99,0,1,0,99)+bytes([255,0,0,255]*3)
    g={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(data),'uri':'data:application/octet-stream;base64,'+base64.b64encode(data).decode()}],
       'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':48,'byteStride':16},{'buffer':0,'byteOffset':48,'byteLength':12}],
       'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},
                    {'bufferView':1,'componentType':5121,'normalized':True,'count':3,'type':'VEC4'}],
       'meshes':[{'primitives':[{'attributes':{'POSITION':0,'COLOR_0':1}}]}],
       'nodes':[{'mesh':0,'translation':[2,3,4]}],'scenes':[{'nodes':[0]}],'scene':0}
    path=tmp_path/'triangle.gltf';path.write_text(json.dumps(g));scene=load_scene(path)
    np.testing.assert_allclose(scene.primitives[0].vertices[0,:3],[2,3,4])
    np.testing.assert_allclose(scene.primitives[0].vertices[:,14:18],[[1,0,0,1]]*3)
    g['extensionsRequired']=['KHR_lights_punctual']
    g['extensions']={'KHR_lights_punctual':{'lights':[{'type':'spot','intensity':20,'range':5,
        'spot':{'innerConeAngle':.1,'outerConeAngle':.5}}]}}
    g['nodes'][0]['children']=[1]
    g['nodes'].append({'translation':[1,0,0],'scale':[2,2,2],'extensions':{'KHR_lights_punctual':{'light':0}}})
    path.write_text(json.dumps(g));loaded=load_scene(path)
    assert len(loaded.lights)==1
    light=loaded.lights[0]
    np.testing.assert_allclose(light.position,[3,3,4]);np.testing.assert_allclose(light.direction,[0,0,-1])
    assert light.range==5 and light.intensity==20 and light.kind=='spot'
