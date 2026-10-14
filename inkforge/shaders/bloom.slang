#include "gt7.slang"
#include "print_filter.slang"
Texture2D<float4> hdrSource;
Texture2D<float4> baseSource;
Texture2D<float4> source;
RWTexture2D<float4> destination;
RWTexture2D<float4> finalImage;
SamplerState linearSampler;
uint2 size;
uint2 bloomSize;
int2 direction;
float exposure;
float threshold;
float strength;
float dotSize;

[shader("compute")]
[numthreads(8,8,1)]
void tonemapMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=size))return;
    finalImage[tid.xy]=float4(gt7SDR(hdrSource.Load(int3(tid.xy,0)).rgb*exposure),1);
}

float siteRank(int2 cell) {
    // Nested, well-spaced sites: lowering occupancy increases inter-dot gaps.
    // Keep sites fixed while the halo moves; changing grid frequency would swim.
    uint2 q=uint2(cell)&7;
    uint rank=0;
    for(uint bit=0;bit<3;bit++) {
        uint x=(q.x>>bit)&1,y=(q.y>>bit)&1;
        rank|=((x^y)*2+y)<<(2*(2-bit));
    }
    return (float(rank)+.5)/64.;
}

[shader("compute")]
[numthreads(8,8,1)]
void extractMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=bloomSize))return;
    float3 sum=0;
    for(int y=0;y<4;y++)for(int x=0;x<4;x++) {
        int2 p=min(int2(tid.xy)*4+int2(x,y),int2(size)-1);
        float4 sample=hdrSource.Load(int3(p,0));
        float3 c=sample.rgb*exposure*sample.a;
        float brightness=max(max(c.r,c.g),c.b);
        // A few extreme emissive texels must not fill an entire facade with ink.
        float peakLimit=max(threshold*4,1.);
        float knee=max(threshold*.5,.001);
        float soft=clamp(brightness-threshold+knee,0,2*knee);
        float extracted=max(brightness-threshold,soft*soft/(4*knee));
        float bounded=peakLimit*extracted/(peakLimit+extracted);
        sum+=c/max(brightness,1e-5)*bounded;
    }
    destination[tid.xy]=float4(sum/16,1);
}

[shader("compute")]
[numthreads(8,8,1)]
void blurMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=bloomSize))return;
    float3 sum=0;float weight=0;
    for(int i=-10;i<=10;i++) {
        float w=exp(-float(i*i)/32.);
        int2 p=clamp(int2(tid.xy)+direction*i,0,int2(bloomSize)-1);
        sum+=source.Load(int3(p,0)).rgb*w;weight+=w;
    }
    destination[tid.xy]=float4(sum/weight,1);
}

[shader("compute")]
[numthreads(8,8,1)]
void compositeMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=size))return;
    float3 base=baseSource.Load(int3(tid.xy,0)).rgb;
    float3 bloomDelta=0;
    if(strength>0) {
        float spacing=max(dotSize,2.);
        float2 p=(float2(tid.xy)+.5)/spacing;
        p=float2(p.x+p.y,p.y-p.x)*.70710678;
        int2 cell=int2(floor(p));
        float2 center=float2(cell)+.5;
        float2 centerPixel=float2(center.x-center.y,center.x+center.y)*.70710678*spacing;
        // One intensity per dot gives a coherent circle, including at halo edges.
        float3 bloom=source.SampleLevel(linearSampler,centerPixel/float2(size),0).rgb;
        float intensity=max(max(bloom.r,bloom.g),bloom.b);
        float tone=(1-exp(-intensity*1.2))*sqrt(saturate(strength/.04));
        float density=sqrt(tone);
        float rank=siteRank(cell);
        float activation=smoothstep(rank,min(rank+.12,1.),density);
        float radius=.30*sqrt(tone*activation);
        float distance=length(frac(p)-.5);
        float aa=.65/spacing;
        float dots=filteredPrintDisk(distance,radius,aa);
        // Both radius and occupied-site density fade; global strength controls luminosity.
        float3 hue=bloom/max(intensity,1e-5);
        hue=lerp(float3(dot(hue,float3(.2126,.7152,.0722))),hue,.5);
        // Print bounded light coverage onto the GT7-mapped scene. Taking the
        // difference of two HDR tone-map evaluations crushed the tiny dots on
        // bright receivers. The screen blend preserves headroom and color,
        // without adding a continuous veil or increasing the dot footprint.
        float printLight=.40*strength/(.04+strength);
        bloomDelta=(1-saturate(base))*hue*printLight*dots;
    }
    finalImage[tid.xy]=float4(linearToSRGB(saturate(base+bloomDelta)),1);
}
