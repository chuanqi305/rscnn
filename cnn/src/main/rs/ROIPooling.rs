#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation fmap_blob;
rs_allocation roi_blob;
rs_allocation out_blob;

int fmap_width;//26
int fmap_height;//20
int pooled_height;//6
int pooled_width;//6
int channel_count;//256
int roi_count;//300
float spatial_scale;//0.0625
int pool_area;// pooled_height * pooled_width;
int batch_pool_area;// pool_area * channel_count;
int fmap_area;//fmap_width * fmap_height

float __attribute__((kernel)) compute(uint32_t x)//x = 300 * 256 * 6 * 6
{
    int fmap_start = ((x % batch_pool_area) / pool_area) * fmap_area;
    int roi_start = (x / batch_pool_area);

    float4 roi = round(rsGetElementAt_float4(roi_blob, roi_start) * spatial_scale);

    float roi_x1 = roi.x;
    float roi_y1 = roi.y;
    float roi_x2 = roi.z;
    float roi_y2 = roi.w;

    int ph = (x % (pool_area))/pooled_width;//0-6;
    int pw = (x % pooled_width);//0-6

    float roi_w_zoom_rate = (roi_x2>roi_x1+1.0?(roi_x2-roi_x1+1):1.0)/(float)pooled_width;
    float roi_h_zoom_rate = (roi_y2>roi_y1+1.0?(roi_y2-roi_y1+1):1.0)/(float)pooled_height;

    int subROI_x1 = (int)floor(roi_w_zoom_rate * (float)pw);//0-26,from 2*2-->20*26
    int subROI_y1 = (int)floor(roi_h_zoom_rate * (float)ph);//0-20
    int subROI_x2 = (int)ceil(roi_w_zoom_rate * (float)(pw + 1));//
    int subROI_y2 = (int)ceil(roi_h_zoom_rate * (float)(ph + 1));//

    subROI_x1 += (int)roi_x1;//(0-26)-->(26+26)
    subROI_y1 += (int)roi_y1;
    subROI_x2 += (int)roi_x1;
    subROI_y2 += (int)roi_y1;

    subROI_x1 = subROI_x1>0?subROI_x1:0;//clip
    subROI_y1 = subROI_y1>0?subROI_y1:0;
    subROI_x2 = subROI_x2<fmap_width?subROI_x2:fmap_width;
    subROI_y2 = subROI_y2<fmap_height?subROI_y2:fmap_height;

    float maxnum = 0.f;

    for(int i=subROI_y1;i<subROI_y2;i++){
        for(int j=subROI_x1;j<subROI_x2;j++){
            int index = fmap_start +  i * fmap_width + j;
            float num = rsGetElementAt_float(fmap_blob, index);
            maxnum = num > maxnum? num : maxnum;
        }
    }
    return  maxnum;
}

float4 __attribute__((kernel)) compute_vector(uint32_t x)//x = 200 * 6 * 6 * 256
{
    int blocks = channel_count / 4;
    int block_offset = x % blocks;

    int batch_pool_area_blocks = batch_pool_area / 4;
    int roi_start = (x / batch_pool_area_blocks);

    float4 roi = round(rsGetElementAt_float4(roi_blob, roi_start) * spatial_scale);

    float roi_x1 = roi.x;
    float roi_y1 = roi.y;
    float roi_x2 = roi.z;
    float roi_y2 = roi.w;

    int point = (x % batch_pool_area_blocks - block_offset) / blocks;
    int pw = point % pooled_width;
    int ph = point / pooled_width;

    //int ph = (x % (pool_area))/pooled_width;//0-6;
    //int pw = (x % pooled_width);//0-6

    float roi_w_zoom_rate = (roi_x2>roi_x1+1.0?(roi_x2-roi_x1+1):1.0)/(float)pooled_width;
    float roi_h_zoom_rate = (roi_y2>roi_y1+1.0?(roi_y2-roi_y1+1):1.0)/(float)pooled_height;

    int subROI_x1 = (int)floor(roi_w_zoom_rate * (float)pw);//0-26,from 2*2-->20*26
    int subROI_y1 = (int)floor(roi_h_zoom_rate * (float)ph);//0-20
    int subROI_x2 = (int)ceil(roi_w_zoom_rate * (float)(pw + 1));//
    int subROI_y2 = (int)ceil(roi_h_zoom_rate * (float)(ph + 1));//

    subROI_x1 += (int)roi_x1;//(0-26)-->(26+26)
    subROI_y1 += (int)roi_y1;
    subROI_x2 += (int)roi_x1;
    subROI_y2 += (int)roi_y1;

    subROI_x1 = subROI_x1>0?subROI_x1:0;//clip
    subROI_y1 = subROI_y1>0?subROI_y1:0;
    subROI_x2 = subROI_x2<fmap_width?subROI_x2:fmap_width;
    subROI_y2 = subROI_y2<fmap_height?subROI_y2:fmap_height;

    float4 maxnum = 0;

    for(int i=subROI_y1;i<subROI_y2;i++){
        for(int j=subROI_x1;j<subROI_x2;j++){
            int index = (i * fmap_width + j) * blocks + block_offset;
            float4 num = rsGetElementAt_float4(fmap_blob, index);
            maxnum = max(maxnum, num);
        }
    }
    return  maxnum;
}

void __attribute__((kernel)) compute_channel256(uint32_t x)//x = 200 * 6 * 6 * 256
{
    int blocks = channel_count / 4;
    int roi_start = (x / pool_area);

    float4 roi = round(rsGetElementAt_float4(roi_blob, roi_start) * spatial_scale);

    float roi_x1 = roi.x;
    float roi_y1 = roi.y;
    float roi_x2 = roi.z;
    float roi_y2 = roi.w;

    int point = x % pool_area;
    int pw = point % pooled_width;
    int ph = point / pooled_width;

    //int ph = (x % (pool_area))/pooled_width;//0-6;
    //int pw = (x % pooled_width);//0-6

    float roi_w_zoom_rate = (roi_x2>roi_x1+1.0?(roi_x2-roi_x1+1):1.0)/(float)pooled_width;
    float roi_h_zoom_rate = (roi_y2>roi_y1+1.0?(roi_y2-roi_y1+1):1.0)/(float)pooled_height;

    int subROI_x1 = (int)floor(roi_w_zoom_rate * (float)pw);//0-26,from 2*2-->20*26
    int subROI_y1 = (int)floor(roi_h_zoom_rate * (float)ph);//0-20
    int subROI_x2 = (int)ceil(roi_w_zoom_rate * (float)(pw + 1));//
    int subROI_y2 = (int)ceil(roi_h_zoom_rate * (float)(ph + 1));//

    subROI_x1 += (int)roi_x1;//(0-26)-->(26+26)
    subROI_y1 += (int)roi_y1;
    subROI_x2 += (int)roi_x1;
    subROI_y2 += (int)roi_y1;

    subROI_x1 = subROI_x1>0?subROI_x1:0;//clip
    subROI_y1 = subROI_y1>0?subROI_y1:0;
    subROI_x2 = subROI_x2<fmap_width?subROI_x2:fmap_width;
    subROI_y2 = subROI_y2<fmap_height?subROI_y2:fmap_height;

    float4 maxnum = 0;

    for(int n=0;n < 64; n++){
        for(int i=subROI_y1;i<subROI_y2;i++){
            for(int j=subROI_x1;j<subROI_x2;j++){
                int index = (i * fmap_width + j) * blocks + n;
                float4 num = rsGetElementAt_float4(fmap_blob, index);
                maxnum = max(maxnum, num);
            }
        }
        rsSetElementAt_float4(out_blob, maxnum, x * 64 + n);
    }
}

