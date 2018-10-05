#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation OutputData;
rs_allocation InputData;
float coeff;

void __attribute__((kernel)) set_zero_vector4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(OutputData, 0, x, y);;
}

void __attribute__((kernel)) copy_vector4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(OutputData, rsGetElementAt_float4(InputData, x, y), x, y);
}

void __attribute__((kernel)) compute_sum_vector4(uint32_t x, uint32_t y)
{
    float4 data = rsGetElementAt_float4(OutputData, x, y);
    float4 input = rsGetElementAt_float4(InputData, x, y);
    data += input * coeff;
    rsSetElementAt_float4(OutputData, data, x, y);
}

void __attribute__((kernel)) compute_max_vector4(uint32_t x, uint32_t y)
{
    float4 data = rsGetElementAt_float4(OutputData, x, y);
    float4 input = rsGetElementAt_float4(InputData, x, y);
    rsSetElementAt_float4(OutputData, max(data, input), x, y);
}

void __attribute__((kernel)) compute_mul_vector4(uint32_t x, uint32_t y)
{
    float4 data = rsGetElementAt_float4(OutputData, x, y);
    float4 input = rsGetElementAt_float4(InputData, x, y);
    rsSetElementAt_float4(OutputData, data * input, x, y);
}