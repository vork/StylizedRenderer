#include "gt7.slang"
#include "surface.slang"
#include "sky.slang"
#include "penumbra.slang"
static const float PI=3.14159265359;
Texture2D<float4> positions;
Texture2D<float4> normals;
Texture2D<float4> bases;
Texture2D<float4> emissions;
Texture2D<float4> geometryNormals;
Texture2D<float> ao;
Texture2D<float> castVisibilityMap;
Texture2D<float4> hatchTextures[4];
SamplerState hatchSampler;
int hatchDirection;
RWTexture2D<float4> outputImage;
RWTexture2D<float4> hdrImage;
RWTexture2D<float> inkMask;
uint2 size;
float3 eye;
float3 sunDirection;
float3 sunColor;
float4 shadowSplits;
float4x4 cameraView;
float4x4 inverseVP;
float4 style; // amount, outline, halftone, specular size
float4 settings; // exposure EV, AO intensity, ambient intensity, sun intensity
float4 effects; // rim, saturation, fog density, dot frequency
float2 comic; // crosshatching, palette simplification
float outlineWobble;
float hatchDensity;
float penumbraStrength;
float shadowFillLevel;
float3 shadowTint;
float3 skyColor;
float worldPixelScale;
int debugView;
struct SceneLight {float4 positionKind;float4 directionRange;float4 colorIntensity;float4 cones;};
StructuredBuffer<SceneLight> sceneLights;
int sceneLightCount;
int skipLightIndex;

float3 safeNormalize(float3 v) {return v*rsqrt(max(dot(v,v),1e-10));}
int cascadeIndex(float depth) {
    return depth<shadowSplits.x?0:depth<shadowSplits.y?1:depth<shadowSplits.z?2:3;
}
float printShadowCoverage(int2 pixel,float3 position,float3 faceNormal) {
    // Reserve a two-pixel transition for tiny printed marks even when a near
    // cascade's physical PCF footprint becomes smaller than a display pixel.
    // Reject other depth layers so foreground silhouettes do not gain halos.
    float footprint=max(length(position-eye)*worldPixelScale,.0001);
    float sum=0,weights=0;
    for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++) {
        int2 q=clamp(pixel+int2(x,y),0,int2(size)-1);
        float4 tap=positions.Load(int3(q,0));
        float3 delta=tap.xyz-position;
        if(tap.w==0||abs(dot(delta,faceNormal))>footprint*.5||length(delta)>footprint*8)continue;
        float weight=float((3-abs(x))*(3-abs(y)));
        sum+=castVisibilityMap.Load(int3(q,0))*weight;weights+=weight;
    }
    return weights>0?sum/weights:castVisibilityMap.Load(int3(pixel,0));
}
float ggxD(float nh,float rough,float expansion) {
    float a=rough*rough, a2=a*a;
    // Lamure's improved NDF. Clamp to its original peak to avoid singular
    // energy spikes when the expanded denominator approaches zero.
    float den=max((1+expansion*pow(1-a,8))*nh*nh*(a2-1)+1,a2);
    return a2/(PI*den*den);
}
float smithVisibility(float nv,float nl,float r) {
    float a2=r*r*r*r;
    return .5/max(nl*sqrt(nv*nv*(1-a2)+a2)+nv*sqrt(nl*nl*(1-a2)+a2),1e-5);
}
float3 fresnel(float vh,float3 f0) {return f0+(1-f0)*pow(1-vh,5);}
float3 sky(float3 d) {
    float t=saturate(d.y*.5+.5);
    return lerp(float3(.55,.38,.29),skyColor,pow(t,.5));
}
float dotPattern(float2 uv,float2 du,float2 dv,float amount) {
    float2 rotated=float2(uv.x+uv.y,uv.y-uv.x)*.70710678;
    float d=length(frac(rotated)-.5);
    float footprint=max(length(du),length(dv));
    float aa=max(footprint*.65,.035);
    // Darkness controls ink coverage, with clearly resolved dots at default scale.
    float radius=sqrt(saturate(amount)*.78/PI);
    float pattern=1-smoothstep(radius-aa,radius+aa,d);
    // Blend to analytic coverage once the lattice is smaller than a pixel.
    return lerp(pattern,PI*radius*radius,smoothstep(.35,1.,footprint));
}
float halftone(float3 p,float3 n,float3 dx,float3 dy,float amount) {
    p*=effects.w; dx*=effects.w;dy*=effects.w;
    float3 weights=pow(abs(n),8); weights/=max(dot(weights,float3(1)),1e-5);
    return dot(weights,float3(dotPattern(p.yz,dx.yz,dy.yz,amount),dotPattern(p.xz,dx.xz,dy.xz,amount),dotPattern(p.xy,dx.xy,dy.xy,amount)));
}
float4 sampleHatch(int index,float2 uv,float lod) {
    // Metal cannot express NonUniformResourceIndex. Select explicitly so each
    // lane may choose a different tonal density without indexing descriptors.
    if(index==0)return hatchTextures[0].SampleLevel(hatchSampler,uv,lod);
    if(index==1)return hatchTextures[1].SampleLevel(hatchSampler,uv,lod);
    if(index==2)return hatchTextures[2].SampleLevel(hatchSampler,uv,lod);
    return hatchTextures[3].SampleLevel(hatchSampler,uv,lod);
}
float hatchLevel(float2 uv,float2 dx,float2 dy,float darkness) {
    float lod=max(0,log2(max(max(length(dx),length(dy))*512.,1.)));
    float level=saturate((darkness-.12)/.88)*3;
    int lower=min(int(floor(level)),3),upper=min(lower+1,3);
    float4 strokes=lerp(sampleHatch(lower,uv,lod),
                        sampleHatch(upper,uv,lod),frac(level));
    int primary=clamp(hatchDirection,0,3);
    int secondary=primary==0?1:primary==1?0:primary==2?3:2;
    float ink=max(strokes[primary],strokes[secondary]*smoothstep(.3,.7,darkness));
    return ink*smoothstep(.08,.25,darkness);
}
float crosshatch(float2 uv,float2 dx,float2 dy,float darkness) {
    float scale=effects.w*.1*hatchDensity;
    uv*=scale;dx*=scale;dy*=scale;
    // Fixed surface frequency: camera motion may filter strokes, never add
    // another octave of them. View-dependent octave blending made AO swim.
    return hatchLevel(uv,dx,dy,darkness);
}

[shader("compute")]
[numthreads(8,8,1)]
void lightingMain(uint3 tid : SV_DispatchThreadID) {
    if(any(tid.xy>=size)) return;
    int2 p=int2(tid.xy);
    float4 pos=positions.Load(int3(p,0));
    inkMask[p]=0;
    float3 color;
    if(pos.w==0) {
        float2 uv=(float2(p)+.5)/float2(size);
        float4 farPoint=mul(inverseVP,float4(uv*float2(2,-2)+float2(-1,1),1,1));
        float3 ray=safeNormalize(farPoint.xyz/farPoint.w-eye);
        color=proceduralSky?illustratedSky(ray):sky(ray)*1.2;
    } else {
        float4 nr=normals.Load(int3(p,0)), bm=bases.Load(int3(p,0)), em=emissions.Load(int3(p,0));
        float4 surface=geometryNormals.Load(int3(p,0));
        float4 gn=float4(decodeNormal(surface.xy),pos.w<0?1:0);
        int2 px=clamp(p+int2(1,0),0,int2(size)-1), py=clamp(p+int2(0,1),0,int2(size)-1);
        float3 dx=positions.Load(int3(px,0)).xyz-pos.xyz,dy=positions.Load(int3(py,0)).xyz-pos.xyz;
        float2 hatchDx=geometryNormals.Load(int3(px,0)).zw-surface.zw;
        float2 hatchDy=geometryNormals.Load(int3(py,0)).zw-surface.zw;
        float3 n=normalize(nr.xyz), v=safeNormalize(eye-pos.xyz), l=sunDirection;
        float3 h=safeNormalize(v+l);
        float nv=max(dot(n,v),.001),nl=max(dot(n,l),0),nh=max(dot(n,h),0),vh=max(dot(v,h),0);
        float r=nr.w, metal=bm.w;
        float3 base=bm.rgb;
        float lum=dot(base,float3(.2126,.7152,.0722));
        base=max(lerp(float3(lum),base,effects.y),0);
        // Compress photographic albedo variation without quantizing speculars.
        float perceptual=sqrt(max(lum,1e-6));
        float bands=floor(perceptual*7)+smoothstep(.42,.58,frac(perceptual*7));
        float flat=pow(max(bands/7,.04),2);
        base*=lerp(1.,flat/max(lum,.002),comic.y*style.x);
        float3 f0=lerp(float3(.04),base,metal), f=fresnel(vh,f0);
        float2 castCoverage=float2(castVisibilityMap.Load(int3(p,0)));
        float castVisibility=castCoverage.y;
        float vis=castVisibility;
        // Back-facing curved receivers must not inherit alternating lit PCF
        // taps from their own front faces. A smooth geometric-normal terminator
        // bounds self-shadow coverage before its narrow printed transition.
        vis=min(vis,lerp(step(0.,dot(gn.xyz,l)),smoothstep(.08,.16,dot(gn.xyz,l)),style.x));
        // Threshold the continuous PCF field, rather than a two-by-two texel
        // comparison. The latter magnifies its raster staircase into sawtooth
        // edges when the comic ramp compresses it to a hard boundary.
        float coreVisibility=min(castCoverage.y,smoothstep(.08,.16,dot(gn.xyz,l)));
        float hardShadow=1-smoothstep(.42,.58,coreVisibility);
        // Finish the low diffuse ramp inside the geometric terminator. A second
        // threshold just beyond it produced a thin parallel stripe on spheres.
        float diffuseBand=.12+.53*smoothstep(.02,.08,nl)+.35*smoothstep(.61,.64,nl);
        float diffuseLight=lerp(nl,diffuseBand,style.x);
        float3 diffuse=(1-f)*(1-metal)*base/PI*diffuseLight;
        float3 spec=f*ggxD(nh,r,style.w*style.x)*smithVisibility(nv,nl,r)*nl;
        // Soft cap only the artistic expansion; reference mode stays standard GGX.
        spec=lerp(spec,min(spec,float3(8)),style.x);
        float visibility=pow(saturate(ao.Load(int3(p,0))*em.w),settings.y);
        // In comic mode crosshatches carry the AO. Retain only a faint
        // contact tone beneath them instead of multiplying two dark layers.
        float printedAO=style.x*saturate(max(style.z,comic.x)*4.);
        float aoBand=lerp(visibility,1.-.08*(1.-visibility),printedAO);
        float hemi=dot(gn.xyz,float3(0,1,0))*.5+.5;
        float3 ambient=lerp(shadowTint,skyColor,hemi*.55)*settings.z;
        float3 envSpec=sky(reflect(-v,n))*fresnel(nv,f0)*lerp(.7,.22,r)*settings.z;
        float printVis=vis;
        if(style.x>0&&penumbraStrength>0) {
            float wider=min(printShadowCoverage(p,pos.xyz,unpackFaceNormal(pos.w)),smoothstep(.08,.16,dot(gn.xyz,l)));
            // A one-pixel caster can disappear in the wider average. Preserve
            // its original partial coverage; broaden only toward a penumbra.
            if(abs(wider-.5)<abs(printVis-.5))printVis=wider;
        }
        float printedVisibility=lerp(vis,1-penumbraInk(surface.zw,hatchDx,hatchDy,1-printVis,hardShadow,penumbraStrength),style.x);
        color=(diffuse+spec)*sunColor*settings.w*printedVisibility*cloudVisibility(pos.xyz,l);
        float shadowFill=lerp(1.,lerp(shadowFillLevel,1.,printedVisibility),style.x);
        color+=((1-metal)*base*ambient+envSpec)*aoBand*shadowFill;
        // Imported punctual lights retain their transforms, inverse-square falloff
        // and spot cones. The selected primary directional light uses the CSM above.
        for(int lightId=0;lightId<sceneLightCount;lightId++) {
            if(lightId==skipLightIndex)continue;
            SceneLight light=sceneLights[lightId];int kind=int(light.positionKind.w);
            if(kind==3)continue;
            float3 vector=light.positionKind.xyz-pos.xyz;
            float distanceSquared=max(dot(vector,vector),.0025);
            float3 ll=kind==0?-light.directionRange.xyz:vector*rsqrt(distanceSquared);
            float attenuation=kind==0?1.:1./distanceSquared;
            if(kind!=0&&light.directionRange.w>0) {
                float ratio=distanceSquared/(light.directionRange.w*light.directionRange.w);
                attenuation*=pow(saturate(1-ratio*ratio),2);
            }
            if(kind==2) {
                float cone=dot(-ll,light.directionRange.xyz);
                attenuation*=pow(saturate((cone-light.cones.y)/max(light.cones.x-light.cones.y,.0001)),2);
            }
            float lnl=max(dot(n,ll),0),lnh=max(dot(n,safeNormalize(v+ll)),0);
            float3 lf=fresnel(max(dot(v,safeNormalize(v+ll)),0),f0);
            float band=.12+.53*smoothstep(.16,.19,lnl)+.35*smoothstep(.61,.64,lnl);
            float diffuseTerm=lerp(lnl,band*smoothstep(0,.08,lnl),style.x);
            float3 ld=(1-lf)*(1-metal)*base/PI*diffuseTerm;
            float3 ls=lf*ggxD(lnh,r,style.w*style.x)*smithVisibility(nv,lnl,r)*lnl;
            ls=lerp(ls,min(ls,float3(8)),style.x);
            color+=(ld+ls)*light.colorIntensity.rgb*light.colorIntensity.w*attenuation;
        }
        // A cool artistic shadow fill keeps large shadow masses readable.
        color+=base*shadowTint*(1-printedVisibility)*.06*style.x*(1-metal);
        float rim=pow(1-nv,4)*smoothstep(-.1,.4,dot(n,l))*(1-r*.5);
        color+=skyColor*rim*effects.x*style.x*visibility;
        float shadowAmount=saturate((1-vis*diffuseLight)*.72+(1-visibility)*.85);
        float dots=halftone(pos.xyz,gn.xyz,dx,dy,shadowAmount);
        float3 ink=color*.08+float3(.008,.009,.018);
        color=lerp(color,ink,dots*style.z*style.x);
        float hatchAmount=saturate((1-visibility)*1.15);
        float hatch=crosshatch(surface.zw,hatchDx,hatchDy,hatchAmount);
        // Neutral near-black pigment, independent of the underlying material.
        // Increase stroke opacity without broadening the filtered coverage.
        float hatchOpacity=1-pow(1-saturate(comic.x),2);
        color=lerp(color,float3(.001,.001,.001),hatch*hatchOpacity*style.x);
        color+=em.rgb;
        if(gn.w>.5) color=base+em.rgb;
        float edge=0;
        // Keep sample locations fixed. Hand-drawn variation modulates pressure;
        // varying the sample radius made silhouettes crawl during camera pans.
        float strokeDistance=length(pos.xyz-eye);
        float strokeFootprint=max(length(dx),length(dy));
        float wobbleFade=exp(-strokeFootprint*8.)/(1.+strokeDistance*strokeDistance*.06);
        float wave=(.65*sin(dot(pos.xyz,float3(2.7,3.1,1.9)))
                   +.35*sin(dot(pos.xyz,float3(7.3,-5.2,6.1))+.7))*wobbleFade;
        float width=max(.25,style.y);
        int inner=int(floor(width));float fraction=frac(width);
        for(int i=0;i<4;i++) {
            int2 direction=i==0?int2(1,0):i==1?int2(-1,0):i==2?int2(0,1):int2(0,-1);
            int2 nearQ=clamp(p+direction,0,int2(size)-1);
            float3 qn=decodeNormal(geometryNormals.Load(int3(nearQ,0)).xy);
            float dist=length(pos.xyz-eye);
            float normalEdge=smoothstep(.28,.65,1-dot(gn.xyz,qn));
            float depthEdge=0;
            for(int tap=0;tap<2;tap++) {
                int2 q=clamp(p+direction*(inner+tap),0,int2(size)-1);
                float4 qp=positions.Load(int3(q,0));
                float planeDistance=abs(dot(qp.xyz-pos.xyz,unpackFaceNormal(pos.w)));
                float e=qp.w==0?1.:smoothstep(.004*dist+.025,.009*dist+.05,planeDistance);
                depthEdge+=e*(tap==0?1-fraction:fraction);
            }
            edge=max(edge,max(depthEdge,normalEdge*.65));
        }
        float pressure=1-min(outlineWobble,.5)*.16*(wave*.5+wobbleFade*.5);
        inkMask[p]=edge*style.x*(style.y>0?1:0);
        color=lerp(color,float3(.012,.016,.035),edge*pressure*style.x*(style.y>0?.88:0));
        float fog=1-exp(-length(pos.xyz-eye)*effects.z);
        color=lerp(color,skyColor*.9,fog);
        if(debugView==1) color=float3(visibility);
        if(debugView==2) color=gn.xyz*.5+.5;
        if(debugView==3) color=base;
        if(debugView==4) color=float3(vis);
        if(debugView==5) color=float3(dots);
        if(debugView==6) color=float3(hatch);
        if(debugView==7) color=float3(halftone(pos.xyz,gn.xyz,dx,dy,.65),crosshatch(surface.zw,hatchDx,hatchDy,.85),0);
        if(debugView==8) {
            const float3 cascadeColors[4]={float3(1,.15,.15),float3(.15,1,.2),float3(.2,.3,1),float3(1,.6,.1)};
            color=cascadeColors[cascadeIndex(-mul(cameraView,float4(pos.xyz,1)).z)]*(.3+.7*vis);
        }
        if(debugView==9) color=float3(castVisibility);
        if(debugView==10) color=float3(smoothstep(.08,.16,dot(gn.xyz,l)));
        if(debugView==11) color=float3(printedVisibility);
    }
    if(debugView==0) {hdrImage[p]=float4(max(color,0),pos.w==0?0:1);return;}
    float3 mapped=saturate(color);
    outputImage[p]=float4(linearToSRGB(mapped),1);
}
