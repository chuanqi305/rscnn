#pragma version(1)
#pragma rs java_package_name(layers)
//#pragma rs_fp_relax
rs_allocation BiasData;

rs_allocation KernelData;
rs_allocation InputData;
rs_allocation ColData;
rs_allocation OutputData;

int inputHeight;// in height
int inputWidth;// in width
int outputHeight;
int outputWidth;

int inputChannel;// in channel
int inputChannelAligned;

int kernelH;// kernel size height
int kernelW;// kernel size width
int padH;
int padW;
int strideH;
int strideW;

int kernelSize;//=kernelH * kernelW
int group;//no support
int dilation;
int relu;

int nblock;// use big vector > 4 (8, 16, 32, 64 ...)

float __attribute__((kernel)) conv_im2col(uint32_t x, uint32_t y)// y = outputHeight * outputWidth,  x = kernelH * kernelW * inputChannel
{
    int outPH = y / outputWidth;
    int outPW = y % outputWidth;
    int kW = (x / inputChannel) % kernelW;
    int kH = (x / inputChannel) / kernelW;
    int c = x % inputChannel;

    int ph = outPH * strideH + kH * dilation - padH;
    if(ph < 0 ||  ph >= inputHeight){
        return 0;
    }

    int pw = outPW * strideW + kW * dilation - padW;
    if (pw < 0 || pw >= inputWidth){
        return 0;
    }

    int idx = ph * inputWidth + pw;
    return rsGetElementAt_float(InputData, c, idx);
}

void __attribute__((kernel)) conv_im2col2(uint32_t x, uint32_t y)// y = outputHeight * outputWidth,  x = kernelH * kernelW
{
    int outPH = y / outputWidth;
    int outPW = y % outputWidth;
    int blocks = inputChannelAligned / 4;

    int kW = x % kernelW;
    int kH = x / kernelW;

    int ph = outPH * strideH + kH * dilation - padH;
    int pw = outPW * strideW + kW * dilation - padW;
    if(ph < 0 || ph >= inputHeight || pw < 0 || pw >= inputWidth){
        for(int i = 0; i < blocks; i++){
            rsSetElementAt_float4(ColData, 0, x * blocks + i, y);
        }
        return;
    }

    int idx = ph * inputWidth + pw;
    float4 value = 0.f;

    for(int i = 0; i < blocks; i++){
        value = rsGetElementAt_float4(InputData, i, idx);
        rsSetElementAt_float4(ColData, value, x * blocks + i, y);
    }
}

float __attribute__((kernel)) conv_bias_relu(float in, uint32_t x, uint32_t y)// x=channel, y=h*w
{
    return max(in + rsGetElementAt_float(BiasData, x), 0.f);
}

float __attribute__((kernel)) conv_bias(float in, uint32_t x, uint32_t y)// x=channel, y=h*w
{
    return in + rsGetElementAt_float(BiasData, x);
}

float __attribute__((kernel)) conv_relu(float in)// x=channel, y=h*w
{
    return max(in, 0.f);
}

void __attribute__((kernel)) conv_dw4(uint32_t x, uint32_t y)// y = outputHeight * outputWidth,  x = inputChannel / 4 = outputChannel / 4
{
    int outPH = y / outputWidth;
    int outPW = y % outputWidth;
    float4 sum = 0.0f;
    float4 fvalue;
    float4 kvalue;
    int c = x * 4;

    for(int h = 0; h < kernelH; h++){
        int ph = outPH * strideH + h * dilation - padH;
        if(ph < 0 ||  ph >= inputHeight){
            continue;
        }
        for(int w = 0; w < kernelW; w++){
            int pw = outPW * strideW + w * dilation - padW;
            if (pw < 0 || pw >= inputWidth){
                continue;
            }
            int idx = ph * inputWidth + pw;
            int kidx = h * kernelW + w;
            fvalue = rsGetElementAt_float4(InputData, x, idx);
            kvalue.x = rsGetElementAt_float(KernelData, kidx, c);
            kvalue.y = rsGetElementAt_float(KernelData, kidx, c + 1);
            kvalue.z = rsGetElementAt_float(KernelData, kidx, c + 2);
            kvalue.w = rsGetElementAt_float(KernelData, kidx, c + 3);
            sum += fvalue * kvalue;
        }
    }
    sum += rsGetElementAt_float4(BiasData, x);
    if(relu){
        sum = max(sum, 0.f);
    }
    rsSetElementAt_float4(OutputData, sum, x, y);
}

void __attribute__((kernel)) conv(uint32_t x, uint32_t y)//x = outputChannel / 4, y = outputHeight * outputWidth,
{
    float4 sum = 0;

    int outPH = y / outputWidth;
    int outPW = y % outputWidth;

    int blocks = inputChannelAligned / 4;

    for (int h = 0 ; h < kernelH ; h++){
        int ph = outPH * strideH + h * dilation - padH;
        if (ph < 0 || ph >= inputHeight)
             continue;
        for (int w = 0 ; w < kernelW ; w++){
            int pw = outPW * strideW + w * dilation - padW;
            if (pw < 0 || pw >= inputWidth)
                continue;
            int idx = ph * inputWidth + pw;
            int kidx = (h * kernelW + w) * blocks;
            for (int i = 0 ; i < blocks ; i++)
            {
                float4 frame_value = rsGetElementAt_float4(InputData, i, idx);

                float4 kernel_value;

                //kernel index x = kernelH * kernelW * inputChannel(group), y = outputChannel
                kernel_value = rsGetElementAt_float4(KernelData,kidx + i , x * 4);
                sum.x += dot(frame_value,kernel_value);

                kernel_value = rsGetElementAt_float4(KernelData,kidx + i, x * 4 + 1);
                sum.y += dot(frame_value,kernel_value);

                kernel_value = rsGetElementAt_float4(KernelData,kidx + i, x * 4 + 2);
                sum.z += dot(frame_value,kernel_value);

                kernel_value = rsGetElementAt_float4(KernelData,kidx + i, x * 4 + 3);
                sum.w += dot(frame_value,kernel_value);
            }
        }
    }
    sum += rsGetElementAt_float4(BiasData, x);
    if(relu){
        sum = max(sum, 0.f);
    }
    rsSetElementAt_float4(OutputData, sum, x, y);
}

void __attribute__((kernel)) conv4n(uint32_t x, uint32_t y)//x = outputChannel / n, y = outputHeight * outputWidth,
{
    float4 sum = 0;

    int outPH = y / outputWidth;
    int outPW = y % outputWidth;

    int blocks = inputChannelAligned / 4;

    int blockGroups = nblock / 4;

    for(int n = 0; n < blockGroups; n++){
        sum = 0;
        int bidx = x * blockGroups + n;
        int xidx = bidx * 4;
        for (int h = 0 ; h < kernelH ; h++){
            int ph = outPH * strideH + h * dilation - padH;
            if (ph < 0 || ph >= inputHeight)
                 continue;
            for (int w = 0 ; w < kernelW ; w++){
                int pw = outPW * strideW + w * dilation - padW;
                if (pw < 0 || pw >= inputWidth)
                    continue;
                int idx = ph * inputWidth + pw;
                int kidx = (h * kernelW + w) * blocks;
                for (int i = 0 ; i < blocks ; i++)
                {
                    float4 frame_value = rsGetElementAt_float4(InputData, i, idx);

                    float4 kernel_value;
                    kernel_value = rsGetElementAt_float4(KernelData,kidx + i , xidx);
                    sum.x += dot(frame_value,kernel_value);

                    kernel_value = rsGetElementAt_float4(KernelData,kidx + i, xidx + 1);
                    sum.y += dot(frame_value,kernel_value);

                    kernel_value = rsGetElementAt_float4(KernelData,kidx + i, xidx + 2);
                    sum.z += dot(frame_value,kernel_value);

                    kernel_value = rsGetElementAt_float4(KernelData,kidx + i, xidx + 3);
                    sum.w += dot(frame_value,kernel_value);
                }
            }
        }
        sum += rsGetElementAt_float4(BiasData, bidx);
        if(relu){
            sum = max(sum, 0.f);
        }
        rsSetElementAt_float4(OutputData, sum, bidx, y);
    }
}


void __attribute__((kernel)) conv1x1(uint32_t x, uint32_t y)//x = outputChannel / 4, y = outputHeight * outputWidth,
{
    float4 sum = 0;
    int blocks = inputChannelAligned / 4;

    float4 frame_value;
    float4 kernel_value;
    int xidx = x * 4;

    for (int i = 0 ; i < blocks ; i++)
    {
        frame_value = rsGetElementAt_float4(InputData, i, y);

        kernel_value = rsGetElementAt_float4(KernelData, i, xidx);
        sum.x += dot(frame_value,kernel_value);

        kernel_value = rsGetElementAt_float4(KernelData, i, xidx + 1);
        sum.y += dot(frame_value,kernel_value);

        kernel_value = rsGetElementAt_float4(KernelData, i, xidx + 2);
        sum.z += dot(frame_value,kernel_value);

        kernel_value = rsGetElementAt_float4(KernelData, i, xidx + 3);
        sum.w += dot(frame_value,kernel_value);
    }
    sum += rsGetElementAt_float4(BiasData, x);
    if(relu){
        sum = max(sum, 0.f);
    }
    rsSetElementAt_float4(OutputData, sum, x, y);
}