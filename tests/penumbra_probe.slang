#include "penumbra.slang"
RWTexture2D<float> result;
float strength;
float2 pan;
float pixelsPerUV;
[shader("compute")]
[numthreads(8,8,1)]
void penumbraTest(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=uint2(256,128)))return;
    float2 pixel=float2(tid.xy)+pan;
    float coverage=pixel.x/255.;
    result[tid.xy]=penumbraInk(pixel/192.,float2(1./192,0),float2(0,1./192),
                             coverage,step(.5,coverage),strength);
}

[shader("compute")]
[numthreads(8,8,1)]
void magnifiedStippleTest(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=uint2(256,128)))return;
    float2 pixel=float2(tid.xy)+.5;
    result[tid.xy]=penumbraInk(pixel/pixelsPerUV,float2(1./pixelsPerUV,0),float2(0,1./pixelsPerUV),.45,0,1);
}

float diskRadius;
[shader("compute")]
[numthreads(8,8,1)]
void diskAreaTest(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=uint2(256,256)))return;
    float2 p=((float2(tid.xy)+.5)/256.-.5)*1.5;
    result[tid.xy]=filteredPrintDisk(length(p),diskRadius,.25);
}
