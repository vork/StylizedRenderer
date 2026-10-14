#include "shadow_sampling.slang"
Texture2D<float2> probeDepth;
RWStructuredBuffer<float4> filterResult;
[shader("compute")]
[numthreads(64,1,1)]
void filterTest(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=256)return;
    float texel=29.+float(tid.x)/255.*6.;
    float3 world=float3((texel+.5)/64.*2-1,0,.5);
    float4x4 identity=float4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
    filterResult[tid.x]=float4(filteredShadow(probeDepth,identity,(float3x3)identity,0.,world,float3(0,0,1)),0,0,0);
}
float probeSlope;
[shader("compute")]
[numthreads(64,1,1)]
void grazingPlaneTest(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=256)return;
    float x=(float(tid.x)/255.-.5)/128.;
    float3 world=float3(x,0,.5+probeSlope*x);
    float4x4 identity=float4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
    float2 coverage=shadowCoverage(probeDepth,identity,(float3x3)identity,.00001,
                                  world,normalize(float3(-probeSlope,0,1)));
    filterResult[tid.x]=float4(coverage,0,0);
}
