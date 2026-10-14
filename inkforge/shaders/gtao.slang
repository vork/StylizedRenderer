#include "surface.slang"
// Ground-truth based ambient occlusion: angular horizon search and analytic
// cosine-weighted integration (Jimenez et al. 2016), with a bilateral denoiser.
// This is a standalone GTAO implementation, not the complete XeGTAO pipeline.
Texture2D<float4> positions;
Texture2D<float4> normals;
Texture2D<float> sourceAO;
RWTexture2D<float> outputAO;
uint2 size;
uint2 fullSize;
float4x4 view;
float radius;
float focalPixels;

int2 surfacePixel(int2 p) { return min(p*2+1,int2(fullSize)-1); }

float3 viewPosition(int2 p) { return mul(view,float4(positions.Load(int3(p,0)).xyz,1)).xyz; }

[shader("compute")]
[numthreads(8,8,1)]
void gtaoMain(uint3 tid : SV_DispatchThreadID) {
    if(any(tid.xy>=size)) return;
    int2 outPixel=int2(tid.xy),p=surfacePixel(outPixel);
    if(positions.Load(int3(p,0)).w==0) {outputAO[outPixel]=1;return;}
    float3 pos=viewPosition(p);
    float3 n=normalize(mul((float3x3)view,decodeNormal(normals.Load(int3(p,0)).xy)));
    float3 v=normalize(-pos);
    // Pull the sample centre slightly forward to avoid self occlusion.
    pos+=v*max(.001,abs(pos.z)*.00008);
    float rp=clamp(radius*focalPixels/max(-pos.z,.01),2.,160.);
    // Fixed quadrature avoids screen-pixel noise crawling over a static surface.
    float noise=.375;
    float visibility=0;
    const int slices=4, steps=7;
    for(int s=0;s<slices;s++) {
        float phi=(s+noise)*3.14159265/slices;
        float2 dir=float2(cos(phi),sin(phi));
        float3 direction=float3(dir.x,-dir.y,0);
        float3 ortho=direction-dot(direction,v)*v;
        float3 axis=normalize(cross(ortho,v));
        float3 pn=n-axis*dot(n,axis);
        float plen=max(length(pn),1e-5);
        float cn=clamp(dot(pn,v)/plen,0,1);
        float angle=sign(dot(ortho,pn))*acos(cn);
        float2 low=cos(float2(angle+1.5707963,angle-1.5707963));
        float2 horizon=low;
        for(int j=0;j<steps;j++) {
            float k=(j+.45+noise*.3)/steps;
            float2 offset=round(dir*(1.5+k*k*rp));
            for(int side=0;side<2;side++) {
                int2 q=p+int2(offset)*(side==0?1:-1);
                if(any(q<0)||any(q>=int2(fullSize))) continue;
                if(positions.Load(int3(q,0)).w==0) continue;
                float3 delta=viewPosition(q)-pos;
                float dist=length(delta);
                float weight=saturate((radius-dist)/(radius*.65));
                float h=dot(delta/max(dist,1e-5),v);
                horizon[side]=max(horizon[side],lerp(low[side],h,weight));
            }
        }
        float2 h=float2(-acos(clamp(horizon.y,-1,1)),acos(clamp(horizon.x,-1,1)));
        // Integral of cos(theta - n) * sin(theta) over each visible arc.
        float2 integral=(cn+2*h*sin(angle)-cos(2*h-angle))*.25;
        visibility+=plen*(integral.x+integral.y);
    }
    outputAO[outPixel]=saturate(visibility/slices);
}

[shader("compute")]
[numthreads(8,8,1)]
void denoiseMain(uint3 tid : SV_DispatchThreadID) {
    if(any(tid.xy>=size)) return;
    int2 outPixel=int2(tid.xy),p=surfacePixel(outPixel);
    if(positions.Load(int3(p,0)).w==0) {outputAO[outPixel]=1;return;}
    float3 pos=positions.Load(int3(p,0)).xyz;
    float3 n=decodeNormal(normals.Load(int3(p,0)).xy);
    float sum=0, weights=0;
    for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++) {
        int2 samplePixel=clamp(outPixel+int2(x,y),0,int2(size)-1);
        int2 q=surfacePixel(samplePixel);
        float4 qp=positions.Load(int3(q,0));
        if(qp.w==0) continue;
        float3 d=qp.xyz-pos;
        float w=exp(-.35*(x*x+y*y)-abs(dot(d,n))*40/max(radius,.01));
        w*=pow(saturate(dot(n,decodeNormal(normals.Load(int3(q,0)).xy))),8);
        sum+=sourceAO.Load(int3(samplePixel,0))*w; weights+=w;
    }
    outputAO[outPixel]=sum/max(weights,1e-5);
}

// Joint bilateral reconstruction: no low-resolution AO bleeds across a
// silhouette or onto a different surface. Full-resolution output for lighting.
[shader("compute")]
[numthreads(8,8,1)]
void upsampleMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=fullSize))return;
    int2 p=int2(tid.xy);
    float4 pos=positions.Load(int3(p,0));
    if(pos.w==0){outputAO[p]=1;return;}
    float3 n=decodeNormal(normals.Load(int3(p,0)).xy);
    float2 coord=(float2(p)-1.)*.5;
    int2 base=int2(floor(coord));float2 f=frac(coord);
    float sum=0,weights=0;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        int2 low=clamp(base+int2(x,y),0,int2(size)-1);
        int2 q=surfacePixel(low);float4 qp=positions.Load(int3(q,0));
        if(qp.w==0)continue;
        float3 d=qp.xyz-pos.xyz;
        float3 qn=decodeNormal(normals.Load(int3(q,0)).xy);
        float w=(x==0?1-f.x:f.x)*(y==0?1-f.y:f.y);
        w*=exp(-abs(dot(d,n))*80/max(radius,.01))*pow(saturate(dot(n,qn)),8);
        sum+=sourceAO.Load(int3(low,0))*w;weights+=w;
    }
    outputAO[p]=weights>1e-5?sum/weights:1.;
}
