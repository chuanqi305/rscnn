#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation bottom_data;
rs_allocation bottom_rois;
rs_allocation top_data;

float spatialScale;
int channels;
int height;
int width;
int pooled_height;//7
int pooled_width;//7

int outputDim;// 8
int groupSize;// 7

void __attribute__((kernel)) compute(uint32_t x)
{
//    int pw = index % pooled_width; // output w
//    int ph = (index / pooled_width) % pooled_height;//output h
//    int ctop = (index / pooled_width / pooled_height) % outputDim; // out channel
    int ctop = x % outputDim;
    int pw = (x / outputDim) % pooled_width; // output w
    int ph = (x / pooled_width / outputDim) % pooled_height;//output h
    int n = x / pooled_width / pooled_height / outputDim;// output n

    //int rois_start = n * 5;
    int rois_start = n * 4;

    //int roi_batch_ind = rsGetElementAt_float(bottom_rois, rois_start);// 0
    float roi_start_w = (float)round(rsGetElementAt_float(bottom_rois, rois_start)) * spatialScale;
    float roi_start_h = (float)round(rsGetElementAt_float(bottom_rois, rois_start + 1)) * spatialScale;
    float roi_end_w = (float)(round(rsGetElementAt_float(bottom_rois, rois_start + 2)) + 1.f) * spatialScale;
    float roi_end_h = (float)(round(rsGetElementAt_float(bottom_rois, rois_start + 3)) + 1.f) * spatialScale;

    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    float roi_height = max(roi_end_h - roi_start_h, 0.1);
    
    // Compute w and h at bottom 
    float bin_size_h = roi_height / (float)(pooled_height);
    float bin_size_w = roi_width / (float)(pooled_width);
    
    int hstart = floor((float)(ph) * bin_size_h + roi_start_h);
    int wstart = floor((float)(pw) * bin_size_w + roi_start_w);
    int hend = ceil((float)(ph + 1) * bin_size_h + roi_start_h);
    int wend = ceil((float)(pw + 1) * bin_size_w + roi_start_w);

    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0),width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    
    int gw = pw;
    int gh = ph;
    int c = (ctop * groupSize + gh) * groupSize + gw;
    
    //int bottom_data_start = (roi_batch_ind * channels + c) * height * width;
    //int channel_offset =  (roi_batch_ind * channels + c);
    int channel_offset = c;
    float out_sum = 0;
    for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
            //int bottom_index = h * width + w;
            int bottom_index = h * (width * channels) + w * channels + channel_offset;
            out_sum += rsGetElementAt_float(bottom_data, bottom_index);
        }
    }
    
    float bin_area = (hend - hstart)*(wend - wstart);
    rsSetElementAt_float(top_data, x, is_empty? 0. : out_sum/bin_area);
}
