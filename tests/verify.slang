#include "gt7.slang"
[shader("compute")]
[numthreads(32,1,1)]
void toneTest(uint3 id:SV_DispatchThreadID, StructuredBuffer<float4> input, RWStructuredBuffer<float4> result, uniform uint count) {
    if(id.x<count) result[id.x]=float4(gt7Rec2020(input[id.x].xyz),1);
}
