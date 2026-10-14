#include "print_filter.slang"
// Fixed surface-space sites at every scale, with continuous scale blending.
// Tiny dots stay tiny close to an object; no changing-frequency swimming.
float stippleLevel(float2 uv,float2 dx,float2 dy,float coverage) {
    float2 cell=floor(uv);
    float2 jitter=frac(sin(float2(dot(cell,float2(127.1,311.7)),dot(cell,float2(269.5,183.3))))*43758.5453);
    float2 center=.5+(jitter-.5)*.1;
    // Ranked, fixed sites vary occupancy as well as radius. Sparse coverage
    // leaves wider gaps; centers never move as the penumbra changes intensity.
    float rank=frac(sin(dot(cell,float2(39.346,73.156)))*47453.5453);
    float activation=saturate((coverage-rank*.55)/.45);
    // Even the largest dot fits inside its jittered cell. The old radius .707
    // crossed cell boundaries, clipping circles into square chunks on closeups.
    float radius=.32*sqrt(activation);
    float footprint=max(length(dx),length(dy));
    float aa=max(.015,footprint*.35);
    float dotInk=filteredPrintDisk(length(frac(uv)-center),radius,aa);
    float expectedActivation=(coverage*coverage-pow(max(coverage-.55,0),2)-pow(max(coverage-.45,0),2))/.495;
    float average=3.14159265*.32*.32*saturate(expectedActivation);
    return lerp(dotInk,average,smoothstep(.8,1.4,footprint));
}

float penumbraInk(float2 uv,float2 dx,float2 dy,float coverage,float core,float strength) {
    float edgeFalloff=smoothstep(.18,.45,coverage)*(1-smoothstep(.55,.82,coverage));
    if(edgeFalloff<=0||strength<=0)return core;
    uv*=128.;dx*=128.;dy*=128.;
    // Smallest singular value handles stretched UVs, not just square charts.
    float a=dot(dx,dx),b=dot(dx,dy),c=dot(dy,dy);
    float minor=sqrt(max(.5*(a+c-sqrt(max((a-c)*(a-c)+4*b*b,0))),1e-12));
    float level=clamp(log2(.75/max(minor,1e-6)),0,16);
    float scale=exp2(floor(level));
    float fine=stippleLevel(uv*scale,dx*scale,dy*scale,coverage);
    float finer=stippleLevel(uv*scale*2,dx*scale*2,dy*scale*2,coverage);
    float filtered=lerp(fine,finer,smoothstep(0,1,frac(level)));
    return lerp(core,filtered,saturate(strength)*edgeFalloff);
}
