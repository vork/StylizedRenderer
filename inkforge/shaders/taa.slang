#include "gt7.slang"
Texture2D<float4> currentColor;
Texture2D<float4> positions;
Texture2D<float4> historyColor;
Texture2D<float4> historyWorld;
Texture2D<float> inkMask;
RWTexture2D<float4> resolved;
RWTexture2D<float4> nextWorld;
SamplerState linearSampler;
uint2 size;
float4x4 previousVP;
float4x4 currentVP;
float4x4 inverseVP;
float2 jitter;
float3 eye;
float worldPixelScale;
bool historyValid;
uint historySamples;
bool coverageEnabled;
RWTexture2DArray<float4> stationarySamples;
uint phaseIndex;
bool stationaryReady;

float4 cubicWeights(float t) {
    float t2=t*t,t3=t2*t;
    return float4(-.5*t3+t2-.5*t,1.5*t3-2.5*t2+1,-1.5*t3+2*t2+.5*t,.5*t3-.5*t2);
}
float4 sharpHistory(float2 uv) {
    float2 pixel=uv*float2(size)-.5;
    int2 base=int2(floor(pixel));
    float4 wx=cubicWeights(frac(pixel.x)),wy=cubicWeights(frac(pixel.y));
    float4 result=0,low=float4(1e10),high=float4(-1e10);
    for(int y=0;y<4;y++)for(int x=0;x<4;x++) {
        float4 sample=historyColor.Load(int3(clamp(base+int2(x-1,y-1),0,int2(size)-1),0));
        result+=sample*wx[x]*wy[y];
        if(x>=1&&x<=2&&y>=1&&y<=2){low=min(low,sample);high=max(high,sample);}
    }
    // Limit reconstruction ringing to the nearest samples' actual range.
    return clamp(result,low,high);
}

[shader("compute")]
[numthreads(8,8,1)]
void taaMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=size))return;
    int2 p=int2(tid.xy);
    float4 world=positions.Load(int3(p,0));
    nextWorld[p]=world;
    float3 current=currentColor.Load(int3(p,0)).rgb;
    float inkCoverage=inkMask.Load(int3(p,0));
    // Store each of the eight actual raster samples, not a recursive resolve.
    // A sliding full-cycle box filter cancels the recurring jitter exactly when
    // the camera is still, while lighting/cloud changes keep updating each frame.
    float4 raw=float4(current,inkCoverage);
    stationarySamples[int3(p,phaseIndex)]=raw;
    if(stationaryReady) {
        float4 sum=0;
        [unroll] for(uint phase=0;phase<8;phase++)
            sum+=phase==phaseIndex?raw:stationarySamples[int3(p,phase)];
        resolved[p]=sum/8.;
        if(world.w==0) {
            float4 priorWorld=historyWorld.Load(int3(p,0));
            nextWorld[p]=float4(priorWorld.xyz,0);
        }
        return;
    }
    // Reconstruct the current sample at the display pixel center before
    // accumulation. Averaging raw jittered samples made one-pixel ink strokes
    // alternate between full black and uncovered paper on each Halton phase.
    if(historyValid)
        current=lerp(current,currentColor.SampleLevel(linearSampler,(float2(p)+.5+jitter)/float2(size),0).rgb,inkCoverage);
    float3 result=current;
    if(historyValid) {
        float2 currentUV=(float2(p)+.5)/float2(size);
        float4 ray=mul(inverseVP,float4(currentUV*float2(2,-2)+float2(-1,1),1,1));
        float4 clip=world.w!=0?mul(previousVP,float4(world.xyz,1)):
            mul(previousVP,float4(ray.xyz/ray.w-eye,0));
        // History is resolved on the unjittered pixel lattice. Do not align
        // history to last frame's jitter: that preserves aliasing and shifts edges.
        float2 uv=clip.xy/max(clip.w,1e-6)*float2(.5,-.5)+.5+jitter/float2(size);
        if(clip.w>0&&all(uv>0)&&all(uv<1)) {
            int2 q=clamp(int2(uv*float2(size)),0,int2(size)-1);
            float4 previousSurface=historyWorld.Load(int3(q,0));
            if(coverageEnabled&&world.w==0&&historyColor.Load(int3(q,0)).a>.01&&
               length(previousSurface.xyz-eye)>12.) {
                // A sky sample that temporarily misses a distant wire still
                // needs the wire's finite-depth camera motion. Infinite-depth
                // sky reprojection leaves its accumulated ink stuck on screen.
                float4 previousPoint=mul(previousVP,float4(previousSurface.xyz,1));
                float4 currentPoint=mul(currentVP,float4(previousSurface.xyz,1));
                if(previousPoint.w>0&&currentPoint.w>0) {
                    float2 motion=(previousPoint.xy/previousPoint.w-currentPoint.xy/currentPoint.w)*float2(.5,-.5);
                    uv=currentUV+motion;
                    q=clamp(int2(uv*float2(size)),0,int2(size)-1);
                    previousSurface=historyWorld.Load(int3(q,0));
                }
            }
            // Sky pixels have no current surface. Carry the last contributing
            // surface position as distance metadata while keeping w=0 (sky),
            // so wires against the sky survive phases which miss the wire.
            if(world.w==0)nextWorld[p]=float4(previousSurface.xyz,0);
            float distanceToEye=length((world.w!=0?world.xyz:previousSurface.xyz)-eye);
            float pixelsMoved=length(uv*float2(size)-(float2(p)+.5));
            float sparseCoverage=smoothstep(12.,24.,distanceToEye)*(1-smoothstep(.5,2.,pixelsMoved));
            sparseCoverage*=smoothstep(12.,24.,length(previousSurface.xyz-eye));
            if(!coverageEnabled)sparseCoverage=0;
            float tolerance=.002+distanceToEye*worldPixelScale*1.5;
            float4 background=world;
            // A thin rail may exist in only one jitter phase. Its neighborhood
            // also contains the background: validate that layer instead of
            // throwing away accumulated coverage whenever the rail is sampled.
            if(sparseCoverage>0&&inkCoverage>0) {
                float farthest=dot(world.xyz-eye,world.xyz-eye);
                for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
                    float4 tap=positions.Load(int3(clamp(p+int2(x,y),0,int2(size)-1),0));
                    float d=dot(tap.xyz-eye,tap.xyz-eye);
                    if(tap.w==0){background=tap;farthest=1e30;}
                    else if(d>farthest){farthest=d;background=tap;}
                }
            }
            bool compatible=false;
            for(int sy=-1;sy<=1;sy++)for(int sx=-1;sx<=1;sx++) {
                float4 sampleWorld=historyWorld.Load(int3(clamp(q+int2(sx,sy),0,int2(size)-1),0));
                compatible=compatible||(world.w==0?sampleWorld.w==0:
                    sampleWorld.w!=0&&distance(sampleWorld.xyz,world.xyz)<tolerance);
                compatible=compatible||(sparseCoverage>0&&inkCoverage>0&&
                    (background.w==0?sampleWorld.w==0:sampleWorld.w!=0&&
                    distance(sampleWorld.xyz,background.xyz)<tolerance));
            }
            if(compatible&&all(uv>0)&&all(uv<1)) {
                float3 low=current,high=current;
                float ink=0;
                for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
                    int2 tap=clamp(p+int2(x,y),0,int2(size)-1);
                    float3 c=currentColor.Load(int3(tap,0)).rgb;
                    low=min(low,c);high=max(high,c);
                    ink=max(ink,inkMask.Load(int3(tap,0)));
                }
                float4 prior=historyColor.SampleLevel(linearSampler,uv,0);
                // Repeated bilinear reprojection washes out thin distant lines
                // during a slow pan. Reconstruct their coverage without ringing.
                if(pixelsMoved>.01&&sparseCoverage>0&&max(prior.a,inkCoverage)>.01)
                    prior=lerp(prior,sharpHistory(uv),sparseCoverage);
                float3 history=prior.rgb;
                // Do not clip a temporally covered distant line to plain paper
                // on every jitter phase that misses it. Retention falls away
                // during larger motion; normal disocclusion rejection remains.
                float retention=sparseCoverage*saturate(prior.a*4);
                low=lerp(low,min(low,history),retention);
                high=lerp(high,max(high,history),retention);
                history=clamp(history,low,high);
                float weight=lerp(.95,.75,saturate(pixelsMoved/8));
                weight=lerp(weight,lerp(.97,.86,saturate(pixelsMoved/8)),ink);
                weight=lerp(weight,.96,sparseCoverage*saturate(max(ink,prior.a)*4));
                // Begin with an unbiased running average. High edge retention
                // must not preserve the first jitter sample after a reset.
                weight=min(weight,float(historySamples)/float(historySamples+1));
                // The sky retains temporal silhouette coverage too.
                result=lerp(current,history,weight);
                inkCoverage=lerp(inkCoverage,prior.a,weight);
            }
        }
    }
    resolved[p]=float4(max(result,0),inkCoverage);
}
