#pragma version(1)
#pragma rs java_package_name(layers)

int channels;
int channelAligned;

rs_allocation inBlob;
rs_allocation outBlob;

int hin;
int win;


//float4 -> float
float __attribute__((kernel)) reshape_to_channel2(uint32_t x,uint32_t y)
{
    int xi = y / (hin * win) + x * hin;
    int yi = y % (hin * win);
    return rsGetElementAt_float(inBlob, xi, yi);
}

//float -> float4
float __attribute__((kernel)) reshape_from_channel2(uint32_t x,uint32_t y)
{
    int xi = x >= (channels / 2) ? 0:1;
    int yi = x / 2 * 30 + y;
    return rsGetElementAt_float(inBlob, xi, yi);
}
