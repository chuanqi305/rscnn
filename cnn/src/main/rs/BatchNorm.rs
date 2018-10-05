#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation mean_blob;
rs_allocation reverse_std_blob;// rsqrt(var + eps)
rs_allocation InputData;

float __attribute__((kernel)) compute(float in, uint32_t x, uint32_t y)
{
    float mean = rsGetElementAt_float(mean_blob, x);
    float rstd = rsGetElementAt_float(reverse_std_blob, x);
    return (in - mean) * rstd;
}

void __attribute__((kernel)) compute_vector4(uint32_t x, uint32_t y)
{
    float4 mean = rsGetElementAt_float4(mean_blob, x);
    float4 rstd = rsGetElementAt_float4(reverse_std_blob, x);
    float4 input = rsGetElementAt_float4(InputData, x, y);
    rsSetElementAt_float4(InputData, (input - mean) * rstd, x, y);
}