#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation scale;
rs_allocation bias;
rs_allocation InputData;

float __attribute__((kernel)) compute(float in, uint32_t x, uint32_t y)
{
    float s = rsGetElementAt_float(scale, x);
    float b = rsGetElementAt_float(bias, x);
    return in * s + b;
}

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
    float4 s = rsGetElementAt_float4(scale, x);
    float4 b = rsGetElementAt_float4(bias, x);
    float4 input = rsGetElementAt_float4(InputData, x, y);
    rsSetElementAt_float4(InputData, input * s + b, x, y);
}
