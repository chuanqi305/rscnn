#pragma version(1)
#pragma rs java_package_name(layers)

float scale_ratio;
rs_allocation InputData;
rs_allocation OutputData;

float __attribute__((kernel)) compute(float in)
{
    return in * scale_ratio;
}

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(OutputData, rsGetElementAt_float4(InputData, x, y) * scale_ratio, x, y);
}