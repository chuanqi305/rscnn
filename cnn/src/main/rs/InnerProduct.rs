#pragma version(1)
#pragma rs java_package_name(layers)
//#pragma rs_fp_relaxed

rs_allocation In_Blob;
rs_allocation Out_Blob;
rs_allocation Kernel_Blob;//weight
rs_allocation Bias_Blob;//bias
int c_i;//input c*h*w (has aligned with 4) [256 * 6 * 6]
int w_w;//weight.length/bias.length or input length not aligned [256 * 6 * 6]
int c_o;//bias length or output length, faster-ConvNet is 4096
int thread_group;

int relu;

void __attribute__((kernel)) compute_f4f1(uint32_t x, uint32_t y) //x 0-300 y 0-4096]
{
    float sum = 0;
	int kernel_offset = y * w_w / 4;//kernel start at (kernel is 256 * 6 * 6)
	int frame_offset = x * w_w / 4;//feature map start at (feature map is 256 * 6 * 6)
    int c_i_new = c_i / 4;
    int out_index = x * c_o + y;

    for (int i = 0 ; i < c_i_new ; i++){//loop 64 * 6 * 6
        float4 frame_value = rsGetElementAt_float4(In_Blob, frame_offset++);
        float4 kernel_value = rsGetElementAt_float4(Kernel_Blob, kernel_offset++);
        sum += dot(frame_value, kernel_value);
    }
    sum += rsGetElementAt_float(Bias_Blob, y);
    if(relu){
        sum = max(sum, 0.f);
    }

    rsSetElementAt_float(Out_Blob, sum, out_index);
}

void  __attribute__((kernel)) compute_f8f1(uint32_t x, uint32_t y)
{
    float sum = 0;

	int kernel_offset = y * w_w / 4;
	int frame_offset = x * w_w / 4;
	int c_i_new = c_i / 8;
	int out_index = x * c_o + y;

    for (int i = 0 ; i < c_i_new ; i++){
        float4 frame_value1 = rsGetElementAt_float4(In_Blob, frame_offset);
		float4 frame_value2 = rsGetElementAt_float4(In_Blob, frame_offset + 1);
        float4 kernel_value1 = rsGetElementAt_float4(Kernel_Blob, kernel_offset);
		float4 kernel_value2 = rsGetElementAt_float4(Kernel_Blob, kernel_offset + 1);
        sum += dot(frame_value1, kernel_value1) + dot(frame_value2, kernel_value2);
        frame_offset += 2;
        kernel_offset += 2;
    }
    sum += rsGetElementAt_float(Bias_Blob, y);
    if(relu){
        sum = max(sum, 0.f);
    }
    rsSetElementAt_float(Out_Blob, sum, out_index);
}

void __attribute__((kernel)) compute_f8fn(uint32_t x, uint32_t y)
{
    int c_i_new = c_i / 8;
    for(int n=0; n<thread_group; n++){
        float sum = 0;
        int kernel_num = y * thread_group + n;
        int kernel_offset = kernel_num * w_w / 4;
        int frame_offset = x * w_w / 4;

        for (int i = 0 ; i < c_i_new ; i++){
            float4 frame_value1 = rsGetElementAt_float4(In_Blob, frame_offset);
            float4 frame_value2 = rsGetElementAt_float4(In_Blob, frame_offset + 1);
            float4 kernel_value1 = rsGetElementAt_float4(Kernel_Blob, kernel_offset);
            float4 kernel_value2 = rsGetElementAt_float4(Kernel_Blob, kernel_offset + 1);
            sum += dot(frame_value1, kernel_value1) + dot(frame_value2, kernel_value2);
            frame_offset += 2;
            kernel_offset += 2;
        }

        sum += rsGetElementAt_float(Bias_Blob, kernel_num);
        if(relu){
            sum = max(sum, 0.f);
        }
        rsSetElementAt_float(Out_Blob, sum, x * c_o + kernel_num);
    }
}

void __attribute__((kernel)) compute_f8fn_1(uint32_t x)
{
    int c_i_new = c_i / 8;
    for(int n=0; n<thread_group; n++){
        int xx = x * thread_group + n ;
        float sum = 0;
        int pic_num = xx / c_o;
        int kernel_num = xx % c_o;
        int kernel_offset = kernel_num * w_w / 4;
        int frame_offset = pic_num * w_w / 4;

        for (int i = 0 ; i < c_i_new ; i++){
            float4 frame_value1 = rsGetElementAt_float4(In_Blob, frame_offset);
            float4 frame_value2 = rsGetElementAt_float4(In_Blob, frame_offset + 1);
            float4 kernel_value1 = rsGetElementAt_float4(Kernel_Blob, kernel_offset);
            float4 kernel_value2 = rsGetElementAt_float4(Kernel_Blob, kernel_offset + 1);
            sum += dot(frame_value1, kernel_value1) + dot(frame_value2, kernel_value2);
            frame_offset += 2;
            kernel_offset += 2;
        }

        sum += rsGetElementAt_float(Bias_Blob, kernel_num);
        if(relu){
            sum = max(sum, 0.f);
        }
        rsSetElementAt_float(Out_Blob, sum, x);
    }
}
