#ifndef INKFORGE_SURFACE_SLANG
#define INKFORGE_SURFACE_SLANG
// Two 12-bit octahedral coordinates fit exactly in a float's integer mantissa.
// Reserve zero for background; all occupied pixels carry a finite positive value.
float2 octSign(float2 v) { return select(v>=0,float2(1),float2(-1)); }
float2 encodeNormal(float3 n) {
    n/=max(dot(abs(n),float3(1)),1e-12);
    return n.z>=0?n.xy:(1-abs(n.yx))*octSign(n.xy);
}
float3 decodeNormal(float2 o) {
    float3 n=float3(o,1-abs(o.x)-abs(o.y));
    if(n.z<0)n.xy=(1-abs(n.yx))*octSign(n.xy);
    return normalize(n);
}
float packFaceNormal(float3 n) {
    n/=max(dot(abs(n),float3(1)),1e-12);
    float2 o=n.z>=0?n.xy:(1-abs(n.yx))*octSign(n.xy);
    uint2 q=uint2(round(saturate(o*.5+.5)*4095));
    return float(q.x+4096*q.y+1);
}
float3 unpackFaceNormal(float value) {
    uint bits=uint(abs(value))-1;
    float2 o=float2(bits%4096,bits/4096)/4095.*2-1;
    float3 n=float3(o,1-abs(o.x)-abs(o.y));
    if(n.z<0) n.xy=(1-abs(n.yx))*octSign(n.xy);
    return normalize(n);
}
#endif
