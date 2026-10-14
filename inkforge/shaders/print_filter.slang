#ifndef INKFORGE_PRINT_FILTER
#define INKFORGE_PRINT_FILTER
// Normalize a radially smoothed disk by its actual integrated area. Multiplying
// by r²/AA² a second time erased tiny pen marks and dim bloom dots.
float filteredPrintDisk(float distance,float radius,float aa) {
    aa=max(aa,1e-6);
    float q=radius/aa;
    float integral=q>=1?radius*radius+.2*aa*aa:
        aa*aa*(4-q)*pow(1+q,4)/40.;
    return saturate((1-smoothstep(radius-aa,radius+aa,distance))*radius*radius/max(integral,1e-12));
}
#endif
