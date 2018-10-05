#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation in_blob;
rs_allocation out_blob;
int pad_h;
int pad_w;
int stride_h;
int stride_w;
int kernel_h;
int kernel_w;
int width;// feature map width
int height;// feature map height
int in_channel;
int out_channel_aligned;// feature map channel align to 4
int feature_map_size;//feature_map_size before pool
int kernel_area; // kernel_H * kernel_W
int pool_width;//feature map height after pool
int pool_height;//feature map height after pool
int pool_size;//pool_width * pool_height

void __attribute__((kernel)) max_pooling_2d(uint32_t x, uint32_t y) //y = outputHeight * outputWidth,  x = outputChannel / 4
{
    float4 max_num = 0.0f;

    int h = ((y / pool_width) * stride_h) - pad_h;
    int w = (y % pool_width) * stride_w - pad_w;

    for(int i=0;i<kernel_w;i++){
        int pw = w + i;
        if(pw<0 || pw >=width){
            continue;
        }
        for(int j=0;j<kernel_h;j++){
            int ph = h + j;
            if(ph < 0 || ph >=height){
                continue;
            }
            int point = ph * width + pw;
            max_num = fmax(rsGetElementAt_float4(in_blob, x, point), max_num);
        }
    }
    rsSetElementAt_float4(out_blob, max_num, x, y);
}

void __attribute__((kernel)) mean_pooling_2d(uint32_t x, uint32_t y) //y = outputHeight * outputWidth,  x = outputChannel
{
    float4 sum = 0.0f;

    int h = ((y / pool_width) * stride_h) - pad_h;
    int w = (y % pool_width) * stride_w - pad_w;

    for(int i=0;i<kernel_w;i++){
        int pw = w + i;
        if(pw<0 || pw >=width){
            continue;
        }
        for(int j=0;j<kernel_h;j++){
            int ph = h + j;
            if(ph < 0 || ph >=height){
                continue;
            }
            int point = ph * width + pw;
            sum += rsGetElementAt_float4(in_blob, x, point);
        }
    }
    rsSetElementAt_float4(out_blob, sum / kernel_area, x, y);
}

void __attribute__((kernel)) global_pooling_2d(uint32_t x) // x = outputChannel
{
    float4 sum = 0.0f;
    int fm_size = height * width;

    for(int i=0;i<fm_size;i++){
        sum += rsGetElementAt_float4(in_blob, x, i);
    }
    rsSetElementAt_float4(out_blob, sum / fm_size, x);
}