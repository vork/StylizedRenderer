"""Deterministic geometry for lighting, material and importer validation."""
import numpy as np
from .scene import Scene, Material, Primitive, make_vertices


def sphere(center, radius=.65, rings=32, sectors=48):
    points=[]; normals=[]; indices=[];uv=[]
    for y in range(rings+1):
        theta=np.pi*y/rings
        for x in range(sectors+1):
            phi=2*np.pi*x/sectors
            n=np.array([np.sin(theta)*np.cos(phi),np.cos(theta),np.sin(theta)*np.sin(phi)])
            normals.append(n);points.append(n*radius+center)
            uv.append([x/sectors,y/rings])
    for y in range(rings):
        for x in range(sectors):
            a=y*(sectors+1)+x;b=a+sectors+1
            indices.extend([a,a+1,b,a+1,b+1,b])
    return make_vertices(points,indices,normals=np.array(normals),uv=np.array(uv))


def plane(size=10,y=0):
    return make_vertices([[-size,y,-size],[-size,y,size],[size,y,size],[size,y,-size]], [0,1,2,0,2,3])


def showroom():
    mats=[Material(name='Paper floor',base=[.65,.58,.42,1],metallic=0,roughness=.85)]
    v,i=plane();prims=[Primitive(v,i,0)]
    colors=[[.78,.04,.025,1],[.015,.32,.6,1],[.92,.44,.03,1],[.3,.06,.55,1]]
    for row in range(2):
        for col in range(4):
            mats.append(Material(name=f'{"Metal" if row else "Dielectric"} r={.12+col*.23:.2f}',base=colors[col],metallic=float(row),roughness=.12+col*.23))
            v,i=sphere(np.array([(col-1.5)*1.7,.7,row*2.1]))
            prims.append(Primitive(v,i,len(mats)-1))
    return Scene(prims,mats,[],'procedural material showroom')


def export_usd(scene,path):
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt
    stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    mats=[]
    for index,m in enumerate(scene.materials):
        material=UsdShade.Material.Define(stage,f'/Materials/M{index}')
        shader=UsdShade.Shader.Define(stage,f'/Materials/M{index}/Preview')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m.base[:3]))
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(m.metallic)
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(m.roughness)
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        mats.append(material)
    for index,p in enumerate(scene.primitives):
        mesh=UsdGeom.Mesh.Define(stage,f'/World/Mesh{index}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p.vertices[:,:3])))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p.vertices[:,3:6])))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(p.vertices[:,6:8])))
        mesh.CreateFaceVertexCountsAttr([3]*(len(p.indices)//3))
        mesh.CreateFaceVertexIndicesAttr(p.indices.astype(int).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[p.material])
    stage.GetRootLayer().Save()
