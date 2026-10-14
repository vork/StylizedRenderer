#include "surface.slang"
// RG stores depth and the actual caster triangle normal (exact packed integer).
// Returns (crisp bilinear core, continuous four-texel penumbra coverage).
float2 shadowCoverage(Texture2D<float2> depthMap,float4x4 matrix,float3x3 normalMatrix,
                     float bias,float3 p,float3 n) {
    float4 q=mul(matrix,float4(p,1));
    float3 ndc=q.xyz/q.w;
    float2 uv=ndc.xy*float2(.5,-.5)+.5;
    if(any(uv<0)||any(uv>1)||ndc.z<0||ndc.z>1)return 1;
    uint w,h;depthMap.GetDimensions(w,h);
    float2 texel=uv*float2(w,h)-.5;
    int2 pixel=int2(floor(texel));float2 phase=frac(texel);
    float3 plane=mul(normalMatrix,n);
    float2 slope=abs(plane.z)>length(plane)*1e-6?-plane.xy/plane.z:float2(0);
    float3x3 inverseLight=transpose(normalMatrix);
    float3 lightDirection=normalize(mul(inverseLight,float3(0,0,-1)));
    float3 receiverNormal=dot(n,lightDirection)<0?-n:n;
    float receiverCos=abs(dot(n,lightDirection));
    float halfTexelWorld=length(mul(inverseLight,float3(1./w,1./h,0)));
    float2 result=0,weightSum=0;
    // Both footprints use the same samples; fractional weights avoid texel jumps.
    for(int y=-2;y<=3;y++)for(int x=-2;x<=3;x++) {
        float2 offset=float2(x,y)-phase;
        float2 weight=float2(saturate(1.-abs(offset.x))*saturate(1.-abs(offset.y)),
                            saturate(2.5-abs(offset.x))*saturate(2.5-abs(offset.y)));
        if(weight.y<=0)continue;
        int2 tap=clamp(pixel+int2(x,y),0,int2(w,h)-1);
        float2 sample=depthMap.Load(int3(tap,0));
        float depth=sample.x;
        float2 tapNdc=(float2(tap)+.5)/float2(w,h)*float2(2,-2)+float2(-1,1);
        float receiver=ndc.z+dot(slope,tapNdc-ndc.xy);
        // On gently curved receivers the neighboring caster triangle has a
        // different tangent plane. Do not let extending the receiver plane
        // beyond that triangle manufacture self-occlusion inside the PCF disk.
        // Real separated blockers retain their depth gap; no normal displacement.
        if(sample.y>0&&receiver-bias>=depth) {
            float3 casterNormal=unpackFaceNormal(sample.y);
            float agreement=abs(dot(n,casterNormal));
            float3 casterPlane=mul(normalMatrix,casterNormal);
            // Nearby facets on a convex surface lie behind one another's
            // tangent planes. They cannot occlude the smooth surface's lit
            // side, even where grazing projection makes depth slopes unstable.
            // Separate parallel layers fail this mutual half-space check.
            if(agreement>.9) {
                casterNormal=dot(casterNormal,lightDirection)<0?-casterNormal:casterNormal;
                float3 delta=mul(inverseLight,float3(tapNdc-ndc.xy,depth-ndc.z));
                float projectedRadius=length(mul(inverseLight,float3(tapNdc-ndc.xy,0)))+halfTexelWorld;
                float support=projectedRadius/max(min(receiverCos,abs(dot(casterNormal,lightDirection))),.05);
                // Include the texel support in the packed-normal/raster
                // precision allowance, including a sample near the query point.
                float tolerance=(length(delta)+halfTexelWorld)*.003;
                bool convex=dot(receiverNormal,delta)<=tolerance&&dot(casterNormal,-delta)<=tolerance;
                if(convex&&length(delta)<=support*1.5) {
                    result+=weight;weightSum+=weight;continue;
                }
            }
            if(agreement>.65&&abs(casterPlane.z)>length(casterPlane)*1e-6) {
                float2 casterSlope=-casterPlane.xy/casterPlane.z;
                // Include the texel's own half-width, not just the offset to its
                // center. The norm bounds curvature in every tangent direction.
                float allowance=length(slope-casterSlope)*(length(tapNdc-ndc.xy)+length(1./float2(w,h)));
                // Curvature tolerance follows the actual filter footprint;
                // retain the full bound on moderately folded cloth surfaces.
                receiver-=allowance*smoothstep(.65,.8,agreement);
            }
        }
        result+=((receiver-bias<depth)?1.:0.)*weight;weightSum+=weight;
    }
    return result/max(weightSum,1e-5);
}
float filteredShadow(Texture2D<float2> depthMap,float4x4 matrix,float3x3 normalMatrix,
                     float bias,float3 p,float3 n) {
    return shadowCoverage(depthMap,matrix,normalMatrix,bias,p,n).y;
}
