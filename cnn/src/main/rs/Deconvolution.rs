#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation KernelData;
rs_allocation BiasData;
rs_allocation InputData;
rs_allocation OutputData;

int padH;
int padW;
int strideH;
int strideW;
int kernelH;
int kernelW;
int channelAligned;
int inputWidth;
int inputHeight;
int outputHeight;
int outputWidth;

void __attribute__((kernel)) deconv_dw4(uint32_t x, uint32_t y)// x = channel / 4
{
    float4 sum = 0;
    int outPH = y / outputWidth;
    int outPW = y % outputWidth;

    for(int j = 0; j < kernelH; j++){
        //int h = outPH + (kernelH - 1 - j) - (kernelH - 1 - padH);
        int h = outPH - j + padH;
        if (h % strideH != 0) {
            continue;
        }
        h /= strideH;
        if(h < 0 || h >= inputHeight) {
            continue;
        }
        for(int k = 0; k < kernelW; k++){
            int w = outPW - k + padW;
            if (w % strideW != 0) {
                continue;
            }
            w /= strideW;
            if (w < 0 || w >= inputWidth){
                continue;
            }
            float4 frame_value = rsGetElementAt_float4(InputData, x, h * inputWidth + w);
            float4 kernel_value = rsGetElementAt_float4(KernelData, x, j * kernelW + k);
            sum += kernel_value * frame_value;
        }
    }
    rsSetElementAt_float4(OutputData,  sum + rsGetElementAt_float4(BiasData, x), x, y);
}