#pragma version(1)
#pragma rs java_package_name(layers)

int channel;
int channelAligned;

rs_allocation inBlob;
rs_allocation expSum;

void __attribute__((kernel)) compute_exp(uint32_t x)// 0-channel
{
    float in;
    if(channelAligned!=channel){
        int channel_offset = x % channel;
        int channel_index = (x - (x % channel))/channel;
        int index = channel_index * channelAligned + channel_offset;
        in = rsGetElementAt_float(inBlob, index);
    }
    else{
        in = rsGetElementAt_float(inBlob, x);
    }
    rsSetElementAt_float(inBlob, exp(in), x);
}

void __attribute__((kernel)) compute_exp_sum(uint32_t x)//
{
    int index = x * channel;
    float sum = 0;
    for(int i=0;i<channel;i++){// skip the aligned
        sum += rsGetElementAt_float(inBlob, index + i);
    }
    rsSetElementAt_float(expSum, sum, x);
}

void __attribute__((kernel)) compute(uint32_t x)// 0 - channel
{
    int index = x / channel;
    float expx = rsGetElementAt_float(inBlob, x);
    float sum = rsGetElementAt_float(expSum, index);
    if(channelAligned!=channel){
        int channel_offset = x % channel;
        int channel_index = (x - (x % channel))/channel;
        int index = channel_index * channelAligned + channel_offset;
        rsSetElementAt_float(inBlob, expx / sum, index);
    }
    else{
        rsSetElementAt_float(inBlob, expx / sum, x);
    }
}
