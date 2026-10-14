#include "surface.slang"
// Static mesh rasterization. All shading executes as native Slang GPU code.
struct Vertex {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 tangent : TANGENT;
    float4 color : COLOR;
    float2 hatchUV : TEXCOORD2;
};
struct Varying {
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 tangent : TANGENT;
    float4 color : COLOR;
    float2 hatchUV : TEXCOORD2;
};
uniform float4x4 viewProjection;
uniform float4 baseFactor;
uniform float4 materialParams; // metallic, roughness, normal scale, occlusion strength
uniform float4 emission; // RGB, unlit
uniform float2 alphaParams; // mode (opaque/mask/blend), cutoff
uniform float textureBias;
StructuredBuffer<float4> triangleNormals;
Texture2D<float4> baseTexture;
Texture2D<float4> mrTexture;
Texture2D<float4> normalTexture;
Texture2D<float4> aoTexture;
Texture2D<float4> emissiveTexture;
SamplerState texSamplers[5];
uniform float4 uvRows[10]; // xyz affine row; w selects UV set

float2 getUV(Varying v, int slot) {
    float2 uv = uvRows[slot*2].w > .5 ? v.uv1 : v.uv;
    return float2(dot(uvRows[slot*2].xyz, float3(uv,1)), dot(uvRows[slot*2+1].xyz, float3(uv,1)));
}
float3 srgbToLinear(float3 x) { return select(x <= .04045, x/12.92, pow((x+.055)/1.055, 2.4)); }

[shader("vertex")]
Varying vertexMain(Vertex v) {
    Varying o;
    o.position = mul(viewProjection, float4(v.position, 1));
    o.world = v.position; o.normal = v.normal; o.uv = v.uv; o.uv1=v.uv1;
    o.tangent=v.tangent; o.color=v.color;
    o.hatchUV=v.hatchUV;
    return o;
}
void alphaTest(float alpha, float2 pixel) {
    if(alphaParams.x == 1 && alpha < alphaParams.y) discard;
    // Ordered coverage for static translucent surfaces; no refraction/sorting required.
    if(alphaParams.x == 2) {
        const float bayer[16] = {0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5};
        uint2 p = uint2(pixel) % 4;
        if(alpha < (bayer[p.x + p.y*4]+.5)/16.) discard;
    }
}
struct GBuffer {
    float4 position : SV_Target0;
    float4 normal : SV_Target1;
    float4 base : SV_Target2;
    float4 emission : SV_Target3;
    float4 geometryNormal : SV_Target4;
};
GBuffer shadeGeometry(Varying v, bool front,uint primitiveID) {
    // Triangle planes must not depend on finite-difference precision or which
    // helper pixels survive early depth. Keep the exact uploaded facet normal.
    float3 face=triangleNormals[primitiveID].xyz;
    float4 base = baseTexture.SampleBias(texSamplers[0], getUV(v,0),textureBias);
    base *= baseFactor * v.color;
    alphaTest(base.a, v.position.xy);
    float3 n = normalize(v.normal) * (front ? 1 : -1);
    float3 geom = n;
    float3 tn = normalTexture.Sample(texSamplers[2], getUV(v,2)).xyz*2-1;
    tn.xy *= materialParams.z;
    float3 t, b;
    if(dot(v.tangent.xyz,v.tangent.xyz)>.1) {
        t=normalize(v.tangent.xyz-n*dot(v.tangent.xyz,n));
        b=cross(n,t)*v.tangent.w;
    } else {
        float3 dx=ddx(v.world), dy=ddy(v.world);
        float2 du=ddx(getUV(v,2)), dv=ddy(getUV(v,2));
        float3 p1=cross(dy,n), p2=cross(n,dx);
        t=p1*du.x+p2*dv.x; b=p1*du.y+p2*dv.y;
        float inv=rsqrt(max(max(dot(t,t),dot(b,b)),1e-12)); t*=inv; b*=inv;
    }
    if(dot(t,t)>.01) n=normalize(t*tn.x+b*tn.y+n*tn.z);
    float4 mr=mrTexture.Sample(texSamplers[1],getUV(v,1));
    float ao=lerp(1.,aoTexture.Sample(texSamplers[3],getUV(v,3)).r,materialParams.w);
    GBuffer o;
    o.position=float4(v.world,packFaceNormal(face)*(emission.w>.5?-1:1));
    o.normal=float4(n,clamp(materialParams.y*mr.g,.045,1));
    o.base=float4(base.rgb,saturate(materialParams.x*mr.b));
    o.emission=float4(emissiveTexture.Sample(texSamplers[4],getUV(v,4)).rgb*emission.rgb,ao);
    o.geometryNormal=float4(encodeNormal(geom),v.hatchUV);
    return o;
}
uniform int targetIndex;
RWTexture2D<float4> storedNormal;
RWTexture2D<float4> storedBase;
RWTexture2D<float4> storedEmission;
RWTexture2D<float4> storedGeometryNormal;

[shader("fragment")]
float4 depthMain(Varying v) : SV_Target0 {
    alphaTest(baseTexture.SampleBias(texSamplers[0],getUV(v,0),textureBias).a*baseFactor.a*v.color.a,v.position.xy);
    return 0;
}

[shader("fragment")]
[earlydepthstencil]
float4 fragmentStorage(Varying v,bool front:SV_IsFrontFace,uint primitiveID:SV_PrimitiveID):SV_Target0 {
    // A matching alpha-tested depth prepass makes these storage writes safe:
    // hidden fragments fail early depth before touching any G-buffer.
    GBuffer g=shadeGeometry(v,front,primitiveID);
    int2 p=int2(v.position.xy);
    storedNormal[p]=g.normal;storedBase[p]=g.base;
    storedEmission[p]=g.emission;storedGeometryNormal[p]=g.geometryNormal;
    return g.position;
}

[shader("fragment")]
float4 fragmentSingle(Varying v, bool front : SV_IsFrontFace,uint primitiveID:SV_PrimitiveID) : SV_Target0 {
    GBuffer g=shadeGeometry(v,front,primitiveID);
    switch(targetIndex) {
        case 0:return g.position;
        case 1:return g.normal;
        case 2:return g.base;
        case 3:return g.emission;
        default:return g.geometryNormal;
    }
}
[shader("fragment")]
GBuffer fragmentMain(Varying v, bool front : SV_IsFrontFace,uint primitiveID:SV_PrimitiveID) {return shadeGeometry(v,front,primitiveID);}
[shader("fragment")]
float2 shadowMain(Varying v,uint primitiveID:SV_PrimitiveID) : SV_Target0 {
    float3 face=triangleNormals[primitiveID].xyz;
    alphaTest(baseTexture.Sample(texSamplers[0],getUV(v,0)).a*baseFactor.a, v.position.xy);
    return float2(v.position.z,packFaceNormal(face));
}
