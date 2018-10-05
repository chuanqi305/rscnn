#pragma version(1)
#pragma rs java_package_name(layers)
rs_allocation In_Blob;
rs_allocation Out_Blob;
float num;

float __attribute__((kernel)) compute(float4 in, uint32_t x)
{
    float sum = 0;
    for(int i=0;i<100;i++){
        float4 s = (in + i) * in;
        sum += s.x + s.y + s.z + s.w;
        //sum += dot(in+i, in);
    }
    return sum;
}

short __attribute__((kernel)) compute_short(short4 in, uint32_t x)
{
    short sum = 0;
    for(short i=0;i<100;i++){
        short4 s = (in + i) * in;
        sum += s.x + s.y + s.z + s.w;
    }
    return sum;
}

int __attribute__((kernel)) compute_int(int4 in, uint32_t x)
{
    int sum = 0;
    for(int i=0;i<100;i++){
        int4 s = (in + i) * in;
        sum += s.x + s.y + s.z + s.w;
    }
    return sum;
}

void __attribute__((kernel)) compute_test(uint32_t x)
{
    float x1 = rsGetElementAt_float(In_Blob, x * 2);
    float y1 = rsGetElementAt_float(In_Blob, x * 2 + 1);

    rsSetElementAt_float(Out_Blob, x1, x * 2);
     rsSetElementAt_float(Out_Blob, y1, x * 2 + 1);
}

void __attribute__((kernel)) compute_test_index(uint32_t x, uint32_t y)
{
    float3 input = rsGetElementAt_float3(In_Blob, x, y);
    if(x==0 && y==1){
        float sum = dot(input, input);
        rsSetElementAt_float(Out_Blob, sum, 0, 0);
    }
}

float __attribute__((kernel)) reshape(float in)
{
    return in;
}