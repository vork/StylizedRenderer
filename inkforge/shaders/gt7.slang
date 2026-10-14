// Slang port of Polyphony Digital's SIGGRAPH 2025 GT7 sample (ICtCp mode).
// MIT License
// Copyright (c) 2025 Polyphony Digital Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

float3 gt7PQ(float3 v) {
    float3 ym=pow(max(v,0)*.01,.1593017578125);
    return pow((.8359375+18.8515625*ym)/(1+18.6875*ym),78.84375);
}
float3 gt7InversePQ(float3 v) {
    float3 np=pow(saturate(v),1/78.84375);
    return 100*pow(max(np-.8359375,0)/(18.8515625-18.6875*np),1/.1593017578125);
}
float3 gt7ToICtCp(float3 c) {
    float3 lms=mul(float3x3(1688,2146,262,683,2951,462,99,309,3688)/4096,c);
    return mul(float3x3(2048,2048,0,6610,-13613,7003,17933,-17390,-543)/4096,gt7PQ(lms));
}
float3 gt7FromICtCp(float3 c) {
    float3 lms=gt7InversePQ(mul(float3x3(1,.00860904,.11103,1,-.00860904,-.11103,1,.560031,-.320627),c));
    return max(mul(float3x3(3.43661,-2.50645,.0698454,-.79133,1.9836,-.192271,-.0259499,-.0989137,1.12486),lms),0);
}
float gt7Curve(float x) {
    x=max(x,0);
    const float peak=2.5, mid=.538, linear=.444, alpha=.25;
    const float k=(linear-1)/(alpha-1);
    if(x<linear*peak) return lerp(mid*pow(x/mid,1.280),x,smoothstep(0,mid,x));
    return peak*linear+peak*k-peak*k*exp(linear/k-x/(k*peak));
}
float3 gt7Rec2020(float3 c) {
    float3 ucs=gt7ToICtCp(c);
    float3 skewed=float3(gt7Curve(c.x),gt7Curve(c.y),gt7Curve(c.z));
    float3 sucs=gt7ToICtCp(skewed);
    float fade=1-smoothstep(.98,1.16,ucs.x/gt7ToICtCp(float3(2.5)).x);
    float3 scaled=gt7FromICtCp(float3(sucs.x,ucs.yz*fade));
    return .4*min(lerp(skewed,scaled,.6),2.5);
}
float3 gt7SDR(float3 linearSRGB) {
    float3 rec2020=mul(float3x3(.627404,.329283,.0433136,.0690973,.91954,.0113623,.0163914,.0880133,.895595),max(linearSRGB,0));
    float3 result=gt7Rec2020(rec2020);
    return saturate(mul(float3x3(1.660491,-.587641,-.072850,-.124550,1.132900,-.008349,-.018151,-.100579,1.118730),result));
}
float3 linearToSRGB(float3 x) {return select(x<=.0031308,12.92*x,1.055*pow(max(x,0),1/2.4)-.055);}
