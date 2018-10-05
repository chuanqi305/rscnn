#pragma version(1)
#pragma rs java_package_name(layers)
rs_allocation InputData;
rs_allocation OutputData;

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
     rsSetElementAt_float4(OutputData, max(rsGetElementAt_float4(InputData, x, y), 0.f), x, y);
}

float __attribute__((kernel)) compute(float in)
{
    return max(in, 0.f);
}
