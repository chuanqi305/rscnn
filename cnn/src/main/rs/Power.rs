#pragma version(1)
#pragma rs java_package_name(layers)
rs_allocation InputData;

float scale;
float shift;
float power;

float __attribute__((kernel)) compute(float in)
{
    return pow(in, power) * scale + shift;
}

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(InputData, pow(rsGetElementAt_float4(InputData, x, y), power) * scale + shift, x, y);
}