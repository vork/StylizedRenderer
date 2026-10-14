// Compare the Slang port against the unmodified Polyphony reference.
#define main gt7_sample_main
#include "../third_party/gt7_tone_mapping.cpp"
#undef main
int main() {
    GT7ToneMapping mapper; mapper.initializeAsSDR();
    const float inputs[][3]={{0,0,0},{.18f,.18f,.18f},{1,1,1},{2.5f,2.5f,2.5f},
        {100,100,100},{.5f,1.23f,.75f},{12.3f,34.3f,56.9f},{1504.7f,64.51f,.5f},
        {10,0,0},{0,10,0},{0,0,10},{.001f,.001f,.001f}};
    for(auto& input:inputs) {
        float out[3]; mapper.applyToneMapping(input,out);
        printf("%.9g %.9g %.9g\n",out[0],out[1],out[2]);
    }
}
