"""Static glTF 2.0 and OpenUSD ingestion; canonical Y-up, metre world space."""
from dataclasses import dataclass, field
from pathlib import Path
import base64
import io
import warnings
from urllib.parse import unquote
import numpy as np
from PIL import Image
from pygltflib import GLTF2
from .math3d import normalize, node_matrix


@dataclass
class Material:
    name: str = "Default"
    base: list = field(default_factory=lambda: [1., 1., 1., 1.])
    metallic: float = 1.
    roughness: float = 1.
    emissive: list = field(default_factory=lambda: [0., 0., 0.])
    normal_scale: float = 1.
    occlusion: float = 1.
    alpha_mode: str = "OPAQUE"
    cutoff: float = .5
    double_sided: bool = False
    unlit: bool = False
    # slots: base color, metallic-roughness, normal, occlusion, emissive
    textures: list = field(default_factory=lambda: [None] * 5)
    texcoords: list = field(default_factory=lambda: [0] * 5)
    transforms: list = field(default_factory=lambda: [[1., 0., 0., 0., 1., 0.]] * 5)
    wraps: list = field(default_factory=lambda: [(10497, 10497)] * 5)


@dataclass
class Primitive:
    vertices: np.ndarray  # position, normal, UV0, UV1, tangent, color, hatch UV: 20 floats
    indices: np.ndarray
    material: int


@dataclass
class Light:
    kind: str = 'point'
    position: tuple = (0.,0.,0.)
    direction: tuple = (0.,0.,-1.) # direction in which the light emits
    color: tuple = (1.,1.,1.)
    intensity: float = 1.
    range: float = 0.
    inner: float = 0.
    outer: float = np.pi/4
    name: str = ''


@dataclass
class Scene:
    primitives: list
    materials: list
    images: list
    source: str
    lights: list = field(default_factory=list)

    @property
    def bounds(self):
        return (np.min([p.vertices[:, :3].min(0) for p in self.primitives], axis=0),
                np.max([p.vertices[:, :3].max(0) for p in self.primitives], axis=0))

    @property
    def triangle_count(self):
        return sum(len(p.indices)//3 for p in self.primitives)

    def batch(self):
        """Merge static instances by material, reducing Bistro's draw overhead."""
        groups = {}
        for p in self.primitives:
            groups.setdefault(p.material, []).append(p)
        merged = []
        for m, ps in groups.items():
            offset, indices = 0, []
            for p in ps:
                indices.append(p.indices + offset)
                offset += len(p.vertices)
            merged.append(Primitive(np.concatenate([p.vertices for p in ps]),
                                    np.concatenate(indices).astype(np.uint32), m))
        self.primitives = merged
        return self


def make_vertices(pos, indices, normals=None, uv=None, uv1=None, tangents=None, colors=None, matrix=None):
    pos = np.asarray(pos, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.uint32).reshape(-1)
    if normals is None:
        triangles = indices.reshape(-1, 3)
        face = np.cross(pos[triangles[:, 1]]-pos[triangles[:, 0]],
                        pos[triangles[:, 2]]-pos[triangles[:, 0]])
        normals = np.zeros_like(pos)
        for i in range(3):
            np.add.at(normals, triangles[:, i], face)
    v = np.zeros((len(pos), 20), np.float32)
    v[:, :3], v[:, 3:6] = pos, normalize(normals)
    if uv is not None:
        v[:, 6:8] = uv
    v[:, 8:10] = uv1 if uv1 is not None else v[:, 6:8]
    if tangents is not None:
        v[:, 10:14] = tangents
    v[:, 14:18] = 1
    if colors is not None:
        v[:, 14:14+colors.shape[1]] = colors
    if matrix is not None:
        v[:, :3] = pos @ matrix[:3, :3].T + matrix[:3, 3]
        v[:, 3:6] = normalize(v[:, 3:6] @ np.linalg.inv(matrix[:3, :3]))
        if tangents is not None:
            v[:, 10:13] = normalize(v[:, 10:13] @ matrix[:3, :3].T)
        if np.linalg.det(matrix[:3, :3]) < 0:
            indices = indices.reshape(-1, 3)[:, [0, 2, 1]].ravel().copy()
            v[:, 13] *= -1
    # Preserve authored surface parameterization and normalize its approximate
    # world texel density per primitive before material batching. This follows
    # curved UV charts and remains fixed under camera motion.
    tri=indices.reshape(-1,3)
    wp=v[tri,:3]
    if uv is not None:
        coords=np.asarray(uv,np.float32)
        a=coords[tri[:,1]]-coords[tri[:,0]];b=coords[tri[:,2]]-coords[tri[:,0]]
        uv_area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()
    else: uv_area=0
    if uv_area>1e-10:
        world_area=np.linalg.norm(np.cross(wp[:,1]-wp[:,0],wp[:,2]-wp[:,0]),axis=1).sum()
        v[:,18:20]=coords*np.sqrt(world_area/uv_area)
    else:
        # Flat projection fallback; curved meshes should supply an authored UV chart.
        axes=np.argsort(np.ptp(v[:,:3],axis=0))[-2:]
        v[:,18:20]=v[:,axes]
    return v, indices


def uri_bytes(uri, root):
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (root / unquote(uri)).read_bytes()


def load_gltf(path):
    path = Path(path)
    g = GLTF2().load(str(path))
    unsupported = set(g.extensionsRequired or []) - {"KHR_materials_unlit", "KHR_texture_transform", "KHR_materials_emissive_strength", "KHR_mesh_quantization", "KHR_lights_punctual"}
    if unsupported:
        raise ValueError(f"Required glTF extensions unsupported: {sorted(unsupported)}. Export uncompressed core glTF.")
    if g.animations or g.skins:
        warnings.warn("Static loader: animation is not evaluated; skinned meshes use their authored rest geometry.")
    buffers = [uri_bytes(b.uri, path.parent) if b.uri else g.binary_blob() for b in g.buffers]
    dtypes = {5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
    widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

    def view_array(view_id, offset, count, width, dtype):
        view = g.bufferViews[view_id]
        dt = np.dtype(dtype)
        stride = view.byteStride or dt.itemsize * width
        return np.ndarray((count, width), dt, buffer=buffers[view.buffer],
                          offset=(view.byteOffset or 0) + offset,
                          strides=(stride, dt.itemsize)).copy()

    def accessor(index):
        a = g.accessors[index]
        width, dt = widths[a.type], dtypes[a.componentType]
        data = (view_array(a.bufferView, a.byteOffset or 0, a.count, width, dt)
                if a.bufferView is not None else np.zeros((a.count, width), dt))
        if a.sparse:
            s = a.sparse
            ids = view_array(s.indices.bufferView, s.indices.byteOffset or 0, s.count, 1, dtypes[s.indices.componentType]).ravel()
            data[ids] = view_array(s.values.bufferView, s.values.byteOffset or 0, s.count, width, dt)
        if a.normalized and a.componentType != 5126:
            maxval = np.iinfo(data.dtype).max
            data = np.maximum(data.astype(np.float32) / maxval, -1.)
        return data

    images = []
    for im in g.images:
        if im.uri:
            data = uri_bytes(im.uri, path.parent)
        else:
            bv = g.bufferViews[im.bufferView]
            data = buffers[bv.buffer][(bv.byteOffset or 0):(bv.byteOffset or 0)+bv.byteLength]
        images.append(data)
    materials = []
    for m in g.materials:
        p = m.pbrMetallicRoughness
        mat = Material(name=m.name or "Material", base=p.baseColorFactor if p else [1, 1, 1, 1],
                       metallic=p.metallicFactor if p else 1., roughness=p.roughnessFactor if p else 1.,
                       emissive=list(np.array(m.emissiveFactor or [0, 0, 0]) * m.extensions.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1)),
                       alpha_mode=m.alphaMode or "OPAQUE", cutoff=m.alphaCutoff if m.alphaCutoff is not None else .5,
                       double_sided=bool(m.doubleSided), unlit="KHR_materials_unlit" in m.extensions,
                       normal_scale=m.normalTexture.scale if m.normalTexture else 1.,
                       occlusion=m.occlusionTexture.strength if m.occlusionTexture else 1.)
        infos = [p.baseColorTexture if p else None, p.metallicRoughnessTexture if p else None,
                 m.normalTexture, m.occlusionTexture, m.emissiveTexture]
        for slot, info in enumerate(infos):
            if info is None:
                continue
            t = g.textures[info.index]
            mat.textures[slot] = t.source
            mat.texcoords[slot] = info.texCoord or 0
            if t.sampler is not None:
                sm = g.samplers[t.sampler]
                mat.wraps[slot] = (sm.wrapS or 10497, sm.wrapT or 10497)
            tr = info.extensions.get("KHR_texture_transform", {})
            sc, off, r = tr.get("scale", [1, 1]), tr.get("offset", [0, 0]), tr.get("rotation", 0)
            c, s = np.cos(r), np.sin(r)
            mat.transforms[slot] = [c*sc[0], -s*sc[1], off[0], s*sc[0], c*sc[1], off[1]]
            mat.texcoords[slot] = tr.get("texCoord", mat.texcoords[slot])
            if mat.texcoords[slot] not in (0, 1):
                raise ValueError("Only TEXCOORD_0 and TEXCOORD_1 are supported")
        materials.append(mat)
    materials.append(Material())
    primitives = []
    lights=[]
    light_defs=(g.extensions or {}).get('KHR_lights_punctual',{}).get('lights',[])

    def visit(node_id, parent):
        node = g.nodes[node_id]
        matrix = parent @ node_matrix(node)
        light_ref=(node.extensions or {}).get('KHR_lights_punctual')
        if light_ref is not None:
            entry=light_defs[light_ref['light']];spot=entry.get('spot',{})
            lights.append(Light(kind=entry['type'],position=tuple(matrix[:3,3]),
                direction=tuple(normalize(matrix[:3,:3]@np.array([0,0,-1]))),
                color=tuple(entry.get('color',[1,1,1])),intensity=float(entry.get('intensity',1)),
                range=float(entry.get('range',0)),inner=spot.get('innerConeAngle',0),
                outer=spot.get('outerConeAngle',np.pi/4),name=entry.get('name','')))
        if node.mesh is not None:
            for p in g.meshes[node.mesh].primitives:
                if p.mode not in (None, 4):
                    raise ValueError(f"Primitive mode {p.mode} is unsupported; triangulate the asset")
                def attr(name):
                    i = getattr(p.attributes, name, None)
                    return accessor(i) if i is not None else None
                pos = attr('POSITION')
                idx = accessor(p.indices).ravel() if p.indices is not None else np.arange(len(pos), dtype=np.uint32)
                verts, idx = make_vertices(pos, idx, attr('NORMAL'), attr('TEXCOORD_0'), attr('TEXCOORD_1'), attr('TANGENT'), attr('COLOR_0'), matrix)
                primitives.append(Primitive(verts, idx, p.material if p.material is not None else len(materials)-1))
        for child in node.children or []:
            visit(child, matrix)
    roots = g.scenes[g.scene or 0].nodes if g.scenes else list(set(range(len(g.nodes)))-{c for n in g.nodes for c in n.children or []})
    for root in roots:
        visit(root, np.eye(4, dtype=np.float32))
    if not primitives:
        raise ValueError("Scene contains no triangle meshes")
    return Scene(primitives, materials, images, str(path),lights).batch()


def load_usd(path):
    from pxr import Usd, UsdGeom, UsdShade, UsdLux
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Could not open USD: {path}")
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    axis = np.eye(4)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        axis[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    axis[:3, :3] *= scale
    mats, images, prims, material_ids = [Material(metallic=0, roughness=.5)], [], [], {}
    lights=[]

    def material_id(prim):
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not bound:
            return 0
        key = str(bound.GetPath())
        if key in material_ids:
            return material_ids[key]
        shader, _, _ = bound.ComputeSurfaceSource()
        m = Material(name=key, metallic=0, roughness=.5)
        if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
            for name, fieldname in [('diffuseColor', 'base'), ('metallic', 'metallic'), ('roughness', 'roughness'), ('emissiveColor', 'emissive'), ('opacity', 'opacity')]:
                inp = shader.GetInput(name)
                val = inp.Get() if inp else None
                if val is not None:
                    if fieldname == 'base': m.base = list(val) + [m.base[3]]
                    elif fieldname == 'emissive': m.emissive = list(val)
                    elif fieldname == 'opacity': m.base[3] = float(val)
                    else: setattr(m, fieldname, float(val))
            threshold = shader.GetInput('opacityThreshold')
            threshold = threshold.Get() if threshold else 0
            m.cutoff = float(threshold or .5)
            m.alpha_mode = 'MASK' if threshold else ('BLEND' if m.base[3] < 1 else 'OPAQUE')
            for name, slot in [('diffuseColor', 0), ('normal', 2), ('occlusion', 3), ('emissiveColor', 4)]:
                inp = shader.GetInput(name)
                source = inp.GetConnectedSource() if inp else None
                if source:
                    tex = UsdShade.Shader(source[0])
                    if tex.GetIdAttr().Get() == 'UsdUVTexture':
                        asset = tex.GetInput('file').Get()
                        filename = asset.resolvedPath or str(Path(path).parent / asset.path)
                        m.textures[slot] = len(images)
                        # USD UVs are bottom-left; convert to glTF/top-left texture convention.
                        im = Image.open(filename).convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        b = io.BytesIO(); im.save(b, format='PNG'); images.append(b.getvalue())
                        cs = tex.GetInput('sourceColorSpace')
                        if cs and cs.Get() == 'raw' and slot in (0, 4):
                            warnings.warn('USD raw color texture is currently treated as sRGB')
            for name in ('metallic', 'roughness'):
                inp = shader.GetInput(name)
                if inp and inp.HasConnectedSource():
                    warnings.warn('USD connected metallic/roughness is not supported; using scalar fallback')
        else:
            warnings.warn(f'{key}: only UsdPreviewSurface is supported; using default material')
        material_ids[key] = len(mats); mats.append(m)
        return len(mats)-1

    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdLux.LightAPI):
            imageable=UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility()==UsdGeom.Tokens.invisible:continue
            api=UsdLux.LightAPI(prim)
            matrix=axis@np.array(cache.GetLocalToWorldTransform(prim)).T
            def value(name,default):
                attr=prim.GetAttribute('inputs:'+name);v=attr.Get(time) if attr else None
                return default if v is None else v
            color=np.array(value('color',[1,1,1]),float)
            if value('enableColorTemperature',False):
                color*=np.array(UsdLux.BlackbodyTemperatureAsRgb(float(value('colorTemperature',6500))))
            kind='directional' if prim.IsA(UsdLux.DistantLight) else 'dome' if prim.IsA(UsdLux.DomeLight) else 'point'
            intensity=float(value('intensity',1))*2**float(value('exposure',0))
            outer=np.radians(float(value('shaping:cone:angle',90)))
            softness=float(value('shaping:cone:softness',0))
            if kind=='point' and outer<np.pi/2:kind='spot'
            if kind in ('point','spot'):
                # Finite USD emitters are approximated as punctual sources.
                radius=float(value('radius',.5))*scale
                if not value('normalize',False):intensity*=np.pi*radius*radius
                if prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight):
                    warnings.warn(f'{prim.GetPath()}: area light approximated as an unshadowed point light')
            if kind=='dome' and value('texture:file',None):
                warnings.warn(f'{prim.GetPath()}: dome texture is not sampled; using dome color/intensity')
            lights.append(Light(kind=kind,position=tuple(matrix[:3,3]),direction=tuple(normalize(matrix[:3,:3]@np.array([0,0,-1]))),
                color=tuple(color),intensity=intensity,inner=outer*(1-softness),outer=outer,name=str(prim.GetPath())))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(time), dtype=np.float32)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(time))
        ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(time))
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        corners, face_ids, start = [], [], 0
        for face, count in enumerate(counts):
            if face not in holes:
                for j in range(1, count-1):
                    corners.extend((start, start+j, start+j+1)); face_ids.extend((face,)*3)
            start += count
        corners, face_ids = np.array(corners), np.array(face_ids)
        if not len(corners): continue
        vertex_ids = ids[corners]
        def expand(data, interpolation):
            if data is None or len(data) == 0: return None
            a = np.asarray(data, dtype=np.float32)
            if interpolation == 'faceVarying': return a[corners]
            if interpolation == 'uniform': return a[face_ids]
            if interpolation == 'constant': return np.tile(a[0], (len(corners), 1))
            return a[vertex_ids]
        normals = expand(mesh.GetNormalsAttr().Get(time), mesh.GetNormalsInterpolation())
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        uv = expand(st.ComputeFlattened(time), st.GetInterpolation()) if st else None
        indices = np.arange(len(corners), dtype=np.uint32)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            indices = indices.reshape(-1, 3)[:, [0, 2, 1]].ravel()
        matrix = axis @ np.array(cache.GetLocalToWorldTransform(prim)).T
        verts, indices = make_vertices(points[vertex_ids], indices, normals, uv, matrix=matrix)
        mid = material_id(prim)
        subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
        if subsets:
            assigned = np.zeros(len(indices)//3, dtype=bool)
            for subset in subsets:
                chosen = np.isin(face_ids[::3], list(subset.GetIndicesAttr().Get()))
                assigned |= chosen
                prims.append(Primitive(verts, indices.reshape(-1, 3)[chosen].ravel(), material_id(subset.GetPrim())))
            if (~assigned).any():
                prims.append(Primitive(verts, indices.reshape(-1, 3)[~assigned].ravel(), mid))
        else:
            prims.append(Primitive(verts, indices, mid))
    if not prims: raise ValueError('USD contains no visible meshes')
    return Scene(prims, mats, images, str(path),lights).batch()


def load_scene(path):
    suffix = Path(path).suffix.lower()
    if suffix in ('.gltf', '.glb'): return load_gltf(path)
    if suffix in ('.usd', '.usda', '.usdc', '.usdz'): return load_usd(path)
    raise ValueError(f'Unsupported scene format: {suffix}')
