#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation in_blob;
rs_allocation out_blob;
float alpha_divide_n;
float beta;
float k;
int width;
int height;
int channel;
int channelAligned;
int local_size;

void __attribute__((kernel)) within_channel(uint32_t x, uint32_t y)//h, w, c/4
{
    float4 data;
    float4 sum = 0.0f;
    int half_local_size = local_size / 2;

    int w = y % width;
    int h = y / width;

    int startx = max((int)h - half_local_size, 0);
    int endx = min((int)h + half_local_size + 1, height);
    int starty = max((int)w - half_local_size, 0);
    int endy = min((int)w + half_local_size + 1, width);

    for(int i=startx; i<endx; i++){
        for(int j=starty; j<endy; j++){
            int idx = i * width + j;
            data = rsGetElementAt_float4(in_blob, x, idx);
            sum += data * data;
        }
    }
    sum = rsGetElementAt_float4(in_blob, x, y) / powr(k + alpha_divide_n * sum, beta);
    rsSetElementAt_float4(out_blob, sum, x, y);
}

void __attribute__((kernel)) cross_channel(uint32_t x, uint32_t y)//h, w, c
{
    float data;
    float sum = 0.0f;
    int startc = max((int)x - local_size / 2, 0);
    int endc = min((int)x + local_size / 2 + 1, height);

    for(int i=startc; i<endc; i++){
        int idx = x * width + y;
        data = rsGetElementAt_float(in_blob, i, idx);
        sum += data * data;
    }
    sum = rsGetElementAt_float(in_blob, x, y) / powr(k + alpha_divide_n * sum, beta);
    rsSetElementAt_float(out_blob, sum, x, y);
}