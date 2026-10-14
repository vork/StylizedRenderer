#include "shadow_sampling.slang"
Texture2D<float4> positions;
Texture2D<float2> shadowMaps[4];
RWTexture2D<float> outputShadow;
uint2 size;
float4x4 lightMatrices[4];
float3x3 lightNormalMatrices[4];
float4 shadowSplits;
float4x4 cameraView;
float shadowBias;
float2 sampleCascade(float3 p, float3 n, int cascade) {
    return shadowCoverage(shadowMaps[cascade],lightMatrices[cascade],lightNormalMatrices[cascade],shadowBias,p,n);
}
int cascadeIndex(float depth) {
    return depth<shadowSplits.x?0:depth<shadowSplits.y?1:depth<shadowSplits.z?2:3;
}
float2 shadowVisibility(float3 p,float3 n) {
    float depth=-mul(cameraView,float4(p,1)).z;
    if(depth>shadowSplits.w) return 1;
    int c=cascadeIndex(depth);
    float2 visibility=sampleCascade(p,n,c);
    float start=c==0?0:shadowSplits[c-1];
    float blend=smoothstep(lerp(start,shadowSplits[c],.88),shadowSplits[c],depth);
    // Only sample a second cascade inside its overlap. Previously every
    // receiver paid for two complete 36-tap, plane-corrected shadow filters.
    if(blend>0) {
        float2 next=c<3?sampleCascade(p,n,c+1):float2(1.);
        visibility=lerp(visibility,next,blend);
    }
    return visibility;
}

[shader("compute")]
[numthreads(8,8,1)]
void shadowMaskMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=size))return;
    float4 p=positions.Load(int3(tid.xy,0));
    outputShadow[tid.xy]=p.w==0?1.:shadowVisibility(p.xyz,unpackFaceNormal(p.w)).y;
}
