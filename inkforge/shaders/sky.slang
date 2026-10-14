// Procedural illustrated sky; all motion is continuous world-direction advection.
float3 skySunDirection;
float skyDaylight;
float skyTime;
float skyPixelAngle;
float3 skyEye;
float cloudShadowStrength;
bool proceduralSky;

float skyHash(float2 p) {return frac(sin(dot(p,float2(127.1,311.7)))*43758.5453);}
float skyNoise(float2 p) {
    float2 i=floor(p),f=frac(p);f=f*f*(3-2*f);
    return lerp(lerp(skyHash(i),skyHash(i+float2(1,0)),f.x),
                lerp(skyHash(i+float2(0,1)),skyHash(i+1),f.x),f.y);
}
float cloudField(float2 p) {
    return .58*skyNoise(p)+.28*skyNoise(p*2.03+17)+.14*skyNoise(p*4.07-11);
}
// Both sky silhouettes and ground shadows sample this world-space cloud layer.
// The same advection applies to either view, so shadows follow moving clouds.
float2 cloudUV(float3 world) {return world.xz/64.+float2(skyTime*.012,skyTime*.004);}
float cloudVisibility(float3 world,float3 light) {
    if(cloudShadowStrength<=0||light.y<=.02||world.y>=180)return 1;
    float3 hit=world+light*((180.-world.y)/light.y);
    float density=cloudField(cloudUV(hit));
    float coverage=smoothstep(.495,.545,density)*smoothstep(.02,.12,light.y);
    return 1-cloudShadowStrength*coverage;
}
float3 illustratedSky(float3 d) {
    float h=saturate(d.y);
    float day=skyDaylight;
    float sunset=pow(1-saturate(abs(skySunDirection.y)*3),3);
    float3 horizon=lerp(float3(.018,.024,.065),float3(.62,.78,.85),day);
    horizon=lerp(horizon,float3(.9,.27,.12),sunset*.78);
    float3 zenith=lerp(float3(.002,.005,.025),float3(.075,.3,.72),day);
    float height=pow(h,.6);
    float bands=floor(height*5)+smoothstep(.42,.58,frac(height*5));
    float3 c=lerp(horizon,zenith,lerp(height,bands/5.,.3));
    // Stars stay fixed on the celestial sphere. Pixel coverage is analytic.
    float2 starUV=encodeNormal(d)*180.;
    float2 cell=floor(starUV);float rank=skyHash(cell);
    float2 center=.25+.5*float2(skyHash(cell+19),skyHash(cell-7));
    float starAA=max(skyPixelAngle*130.,.04),radius=.20+.14*skyHash(cell+3);
    float star=1-smoothstep(radius-starAA,radius+starAA,length(frac(starUV)-center));
    star*=saturate(radius*radius/(starAA*starAA))*step(.992,rank)*pow(1-day,4)*smoothstep(0,.16,d.y);
    c+=float3(.75,.85,1.2)*star*4.;
    float sunDistance=length(d-skySunDirection);
    float sunDisk=1-smoothstep(.018-skyPixelAngle,.018+skyPixelAngle,sunDistance);
    c+=float3(5.,3.1,1.2)*sunDisk*smoothstep(-.04,0.,skySunDirection.y);
    float3 moonDirection=-skySunDirection;
    float3 moonTangent=normalize(cross(float3(0,1,0),moonDirection)+float3(.00001,0,0));
    float moonDisk=1-smoothstep(.026-skyPixelAngle,.026+skyPixelAngle,length(d-moonDirection));
    float crescent=smoothstep(.024-skyPixelAngle,.024+skyPixelAngle,length(d-normalize(moonDirection+moonTangent*.017)));
    c+=float3(.65,.78,1.)*moonDisk*(.05+.95*crescent)*(1-day)*1.3;
    // A curved cloud height field keeps the illustrated silhouettes, while surface
    // normals, thickness and short light marches give the puffs apparent volume.
    float3 cloudHit=skyEye+d*(max(180.-skyEye.y,1.)/max(d.y,.015));
    float2 uv=cloudUV(cloudHit);
    float density=cloudField(uv);
    float cloudAA=max(.003,skyPixelAngle*2/max(d.y,.04));
    float cover=smoothstep(.51-cloudAA,.51+cloudAA,density)*smoothstep(.015,.14,d.y);
    float3 cloudSun=skySunDirection.y>=0?skySunDirection:-skySunDirection;
    float heightField=sqrt(saturate((density-.49)*3.5));
    float gx=cloudField(uv+float2(.055,0))-cloudField(uv-float2(.055,0));
    float gz=cloudField(uv+float2(0,.055))-cloudField(uv-float2(0,.055));
    float3 puffNormal=normalize(float3(-gx,.027,-gz));
    float illumination=saturate(dot(puffNormal,cloudSun))*.75+.25;
    float2 marchDirection=cloudSun.xz/max(length(cloudSun.xz),.1);
    float lightSlope=cloudSun.y/max(length(cloudSun.xz),.15);
    float occlusion=0;
    for(int i=1;i<=3;i++) {
        float stepDistance=float(i)*.17;
        float otherHeight=sqrt(saturate((cloudField(uv+marchDirection*stepDistance)-.49)*3.5));
        float rayHeight=heightField+stepDistance*lightSlope;
        occlusion=max(occlusion,smoothstep(.015,.13,otherHeight-rayHeight));
    }
    illumination*=1-occlusion*.65;
    // Keep broad cel tones, with a small continuous component for roundness.
    float cel=.15+.45*smoothstep(.3,.37,illumination)+.4*smoothstep(.68,.76,illumination);
    float lightLevel=lerp(cel,illumination,.22);
    float thickness=smoothstep(.54,.72,density);
    lightLevel*=1-thickness*.16;
    float3 cloudShadow=lerp(float3(.025,.035,.085),float3(.32,.48,.69),day);
    float3 cloudLight=lerp(float3(.13,.17,.29),float3(.95,.95,.84),day);
    cloudLight=lerp(cloudLight,float3(1.1,.48,.25),sunset*.7);
    float3 cloud=lerp(cloudShadow,cloudLight,lightLevel);
    float rim=(1-smoothstep(.515,.565,density))*pow(saturate(dot(d,cloudSun)),6);
    cloud+=cloudLight*rim*.28;
    c=lerp(c,cloud,cover);
    return lerp(horizon*.7,c,smoothstep(-.12,.01,d.y));
}
