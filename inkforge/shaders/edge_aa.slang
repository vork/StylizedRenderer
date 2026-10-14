// Spatial edge search/subpixel filtering following Lottes' FXAA. Adapted from
// the user's ntc_viewer implementation for a compute pass on display-linear SDR.
Texture2D<float4> source;
RWTexture2D<float4> destination;
SamplerState linearSampler;
uint2 size;

float3 colorAt(float2 uv) {return source.SampleLevel(linearSampler,uv,0).rgb;}
float lumaAt(float2 uv) {return sqrt(max(dot(colorAt(uv),float3(.299,.587,.114)),0));}
float3 filterEdge(float2 uv) {
    float2 pixel=1./float2(size);
    float3 center=colorAt(uv);
    float m=sqrt(max(dot(center,float3(.299,.587,.114)),0));
    float n=lumaAt(uv+float2(0,-pixel.y)),s=lumaAt(uv+float2(0,pixel.y));
    float w=lumaAt(uv+float2(-pixel.x,0)),e=lumaAt(uv+float2(pixel.x,0));
    float low=min(m,min(min(n,s),min(w,e))),high=max(m,max(max(n,s),max(w,e)));
    float range=high-low;
    if(range<max(.025,high*.10))return center;
    float nw=lumaAt(uv-pixel),ne=lumaAt(uv+pixel*float2(1,-1));
    float sw=lumaAt(uv+pixel*float2(-1,1)),se=lumaAt(uv+pixel);
    float horizontal=abs(nw+sw-2*w)+2*abs(n+s-2*m)+abs(ne+se-2*e);
    float vertical=abs(nw+ne-2*n)+2*abs(w+e-2*m)+abs(sw+se-2*s);
    bool alongX=horizontal>=vertical;
    float negative=alongX?n:w,positive=alongX?s:e;
    bool useNegative=abs(negative-m)>=abs(positive-m);
    float neighbor=useNegative?negative:positive;
    float gradient=abs(neighbor-m),edgeLuma=(m+neighbor)*.5;
    float2 normalStep=(alongX?float2(0,pixel.y):float2(pixel.x,0))*(useNegative?-1:1);
    float2 tangentStep=alongX?float2(pixel.x,0):float2(0,pixel.y);
    float2 edgeUV=uv+normalStep*.5;
    float2 ends[2]={edgeUV,edgeUV};
    float endpointLuma[2]={0,0};bool found[2]={false,false};
    const float strides[12]={1,1,1,1,1.5,2,2,2,2,4,4,8};
    for(uint step=0;step<12;step++) {
        for(uint side=0;side<2;side++)if(!found[side]) {
            ends[side]+=tangentStep*strides[step]*(side==0?-1:1);
            endpointLuma[side]=lumaAt(ends[side])-edgeLuma;
            found[side]=abs(endpointLuma[side])>=gradient*.25;
        }
        if(found[0]&&found[1])break;
    }
    float d0=alongX?uv.x-ends[0].x:uv.y-ends[0].y;
    float d1=alongX?ends[1].x-uv.x:ends[1].y-uv.y;
    uint closest=d0<d1?0:1;
    float edgeBlend=.5-min(d0,d1)/max(d0+d1,1e-6);
    if((endpointLuma[closest]<0)==(m<edgeLuma))edgeBlend=0;
    float average=(2*(n+s+w+e)+nw+ne+sw+se)/12;
    float subpixel=saturate(abs(average-m)/range);
    subpixel=subpixel*subpixel*(3-2*subpixel);
    subpixel=subpixel*subpixel*.75;
    return colorAt(uv+normalStep*max(edgeBlend,subpixel));
}

[shader("compute")]
[numthreads(8,8,1)]
void edgeAAMain(uint3 tid:SV_DispatchThreadID) {
    if(any(tid.xy>=size))return;
    int2 p=int2(tid.xy);
    float4 center=source.Load(int3(p,0));
    float ink=center.a;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++)
        ink=max(ink,source.Load(int3(clamp(p+int2(x,y),0,int2(size)-1),0)).a);
    // Only reconstruct the ink neighborhood. Temporal mask coverage keeps
    // this selection stable; resolved textures and pen stipples stay sharp.
    float3 color=center.rgb;
    if(ink>.01)color=lerp(color,filterEdge((float2(p)+.5)/float2(size)),saturate(ink)*.3);
    destination[p]=float4(color,1);
}
