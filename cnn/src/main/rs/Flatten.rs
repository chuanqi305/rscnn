#pragma version(1)
#pragma rs java_package_name(layers)
rs_allocation InputData;
rs_allocation OutputData;
int channels;
int blocks;

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(OutputData, rsGetElementAt_float4(InputData, x, y), y * blocks + x);
}

void __attribute__((kernel)) compute(float in, uint32_t x, uint32_t y)
{
    if(x < channels){
        rsSetElementAt_float(OutputData, in, y * channels + x);
    }
}
