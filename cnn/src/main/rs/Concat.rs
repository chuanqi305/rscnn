#pragma version(1)
#pragma rs java_package_name(layers)

int blockOffset;
int channelOffset;
int width;
int height;
int outBlockAligned;
int inBlockAligned;
int inChannel;
int inChannelAligned;
int outChannelAligned;

rs_allocation in_Blob;
rs_allocation out_Blob;

int offset;

void __attribute__((kernel)) compute_in4out4(uint32_t x, uint32_t y)
{
    rsSetElementAt_float4(out_Blob, rsGetElementAt_float4(in_Blob, x, y), x + blockOffset, y);
}

void __attribute__((kernel)) compute(uint32_t x, uint32_t y)
{
    rsSetElementAt_float(out_Blob,  rsGetElementAt_float(in_Blob, x, y), x + channelOffset, y);
}