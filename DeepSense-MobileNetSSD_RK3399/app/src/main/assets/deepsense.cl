#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//static inline int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3)
inline int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3)
{
    return i1 * (d2 * d3) + i2 * d3 + i3;
}

//static inline int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4)
inline int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4)
{
    return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

__kernel void convertFloatToHalfBias(
        global const float *input,
        global half *output,
const int bias) {
int idx = get_global_id(0);
vstore_half(input[idx], bias, &output[idx]);
}

__kernel void convertFloatToHalf(
        global const float *input,
        global half *output) {
int idx = get_global_id(0);
vstore_half(input[idx], 0, &output[idx]);
}

__kernel void convertHalfToFloat(
        global const half *input,
global float *output) {
int idx = get_global_id(0);
//output[idx] = convert_float(input[idx]);
output[idx] = (float)input[idx];
}


//无优化的第一层卷积
//first conv
__kernel void conv_first_kernel_half(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c){
int x,y,z,n,i,j;

int threadId_x = get_global_id(0);
int threadId_y = get_global_id(1);
int threadId_z = get_global_id(2);


for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
half result = 0.0f;
for(y = 0 ; y < conv_h ; y++) {
int global_input_y = j * stride_h - pad_top + y;
{
for(x = 0 ; x < conv_w ; x++) {
int global_input_x = i * stride_w - pad_left + x;
if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h)
{
int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, 0);
int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, 0);
half2 tmp_input = vload2(0, &input[global_input_index]);
half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
result += dot(tmp_input, tmp_weight);

result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
}
}
}
}
result += bias[n];
output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
//                output[getIndexFrom3D(output_c, output_h, output_w, n, j, i)] = result > 0 ? result : 0;
}
}
}
}


//first conv
/*
__kernel void conv_first_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                half result = 0.0f;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    {
                        int global_input_x = i * stride_w - pad_left;

                        if(global_input_y >= 0 && global_input_y < input_h)
                        {
                            if( 0 <= global_input_x && global_input_x <= input_w-3 )
                            {
                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, 0, 0);
                                int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, 0);
                                half8 tmp_input = vload8(0, &input[global_input_index]);
                                half8 tmp_weight = vload8(0, &conv_weight[global_filter_index]);
                                result += dot(tmp_input.s0123, tmp_weight.s0123);
                                result += dot(tmp_input.s4567, tmp_weight.s4567);
                                result += input[global_input_index + 8] * conv_weight[global_filter_index + 8];
                            }
                            else if(global_input_x < 0)
                            {
                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, 1, 0);
                                int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x + 1, 0);
                                half4 tmp_input = vload4(0, &input[global_input_index]);
                                half4 tmp_weight = vload4(0, &conv_weight[global_filter_index]);
                                result += dot(tmp_input.s0123, tmp_weight.s0123);
                                half2 tmp_input2 = vload2(0, &input[global_input_index + 4]);
                                half2 tmp_weight2 = vload2(0, &input[global_filter_index + 4]);
                                result += dot(tmp_input2, tmp_weight2);

                                //result += input[global_input_index + 4] * conv_weight[global_filter_index + 4] + input[global_input_index + 5] * conv_weight[global_filter_index + 5];
                            }
                            else if(global_input_x > input_w-3)
                            {
                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, 0, 0);
                                int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, 0);
                                half4 tmp_input = vload4(0, &input[global_input_index]);
                                half4 tmp_weight = vload4(0, &conv_weight[global_filter_index]);
                                result += dot(tmp_input.s0123, tmp_weight.s0123);
                                half2 tmp_input2 = vload2(0, &input[global_input_index + 4]);
                                half2 tmp_weight2 = vload2(0, &input[global_filter_index + 4]);
                                result += dot(tmp_input2, tmp_weight2);

                                //result += input[global_input_index + 4] * conv_weight[global_filter_index + 4] + input[global_input_index + 5] * conv_weight[global_filter_index + 5];
                            }
                        }
                    }
                }

                result += bias[n];

                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
                output[getIndexFrom3D(output_c, output_h, output_w, n, j, i)] = result;
            }
        }
    }
}
*/

//无优化的DW卷积
__kernel void conv_dw_kernel_half(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c) {
int x,y,z,n,i,j;

int threadId_x = get_global_id(0);
int threadId_y = get_global_id(1);
int threadId_z = get_global_id(2);

for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
half result = 0.0f;
for(y = 0 ; y < conv_h ; y++) {
int global_input_y = j * stride_h - pad_top + y;
for(x = 0 ; x < conv_w ; x++) {
int global_input_x = i * stride_w - pad_left + x;
if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
int global_filter_index = getIndexFrom3D(conv_n, conv_h, conv_w, n, y, x);
int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, n);

result += input[global_input_index] * conv_weight[global_filter_index];
}
}
}

result += bias[n];
output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
}
}
}
}


/*
//dw conv
__kernel void conv_dw_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                half result = 0.0f;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    if(global_input_y >= 0 && global_input_y < input_h)
                    {
                        int global_input_x = i * stride_w - pad_left;
                        if(global_input_x < 0)
                        {
                            int global_filter_index = getIndexFrom3D(conv_n, conv_h, conv_w, n, y, 1);
                            int global_input_index = getIndexFrom3D(input_c, input_h, input_w, n, global_input_y, global_input_x + 1);
                            half2 tmp_input = vload2(0, &input[global_input_index]);
                            half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                            result += dot(tmp_input, tmp_weight);
                        }
                        else if(global_input_x > input_w-3)
                        {
                            int global_filter_index = getIndexFrom3D(conv_n, conv_h, conv_w, n, y, 0);
                            int global_input_index = getIndexFrom3D(input_c, input_h, input_w, n, global_input_y, global_input_x);
                            half2 tmp_input = vload2(0, &input[global_input_index]);
                            half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                            result += dot(tmp_input, tmp_weight);
                        }
                        else
                        {
                            int global_filter_index = getIndexFrom3D(conv_n, conv_h, conv_w, n, y, 0);
                            int global_input_index = getIndexFrom3D(input_c, input_h, input_w, n, global_input_y, global_input_x);
                            half2 tmp_input = vload2(0, &input[global_input_index]);
                            half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                            result += dot(tmp_input, tmp_weight);
                            result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
                        }
                    }
                }

                result += bias[n];

                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
            }
        }
    }
}
*/

//1*1 点卷积
__kernel void conv_1_1_kernel_half(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c) {
int x,y,z,n,i,j;

int threadId_x = get_global_id(0);
int threadId_y = get_global_id(1);
int threadId_z = get_global_id(2);

for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
half result = 0.0f;
int global_input_y = j * stride_h - pad_top;
int global_input_x = i * stride_w - pad_left;
for(z = 0 ; z < conv_c ; z += 16) {
int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, 0, 0, z);
int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

half16 tmp_input = vload16(0, &input[global_input_index]);
half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

result += dot(tmp_input.s0123, tmp_weight.s0123);
result += dot(tmp_input.s4567, tmp_weight.s4567);
result += dot(tmp_input.s89ab, tmp_weight.s89ab);
result += dot(tmp_input.scdef, tmp_weight.scdef);

}

result += bias[n];
output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
}
}
}
}


__kernel void conv_1_1_kernel_half_no_relu(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c) {
int x,y,z,n,i,j;

int threadId_x = get_global_id(0);
int threadId_y = get_global_id(1);
int threadId_z = get_global_id(2);

for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
half result = 0.0f;
int global_input_y = j * stride_h - pad_top;
int global_input_x = i * stride_w - pad_left;
for(z = 0 ; z < conv_c ; z += 16) {
int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, 0, 0, z);
int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

half16 tmp_input = vload16(0, &input[global_input_index]);
half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

result += dot(tmp_input.s0123, tmp_weight.s0123);
result += dot(tmp_input.s4567, tmp_weight.s4567);
result += dot(tmp_input.s89ab, tmp_weight.s89ab);
result += dot(tmp_input.scdef, tmp_weight.scdef);
}

result += bias[n];
output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
}
}
}
}

/*
__kernel void conv_1_1_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(1);
    int threadId_y = get_global_id(2);
    int threadId_z = get_global_id(0);


        for(j = threadId_y ; j < output_h ; j += get_global_size(2)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(1)) {
                for(n = threadId_z*4 ; n < output_c ; n+=get_global_size(0)*4) {

                half r[4];
                int global_input_y = j * stride_h - pad_top;
                int global_input_x = i * stride_w - pad_left;

                for(int c=0;c<4;c++)
                {
                    half result = 0.0f;
                for(z = 0 ; z < conv_c ; z += 16) {
                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n+c, 0, 0, z);
                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                    half16 tmp_input = vload16(0, &input[global_input_index]);
                    half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                                result += dot(tmp_input.s0123, tmp_weight.s0123);
                                result += dot(tmp_input.s4567, tmp_weight.s4567);
                                result += dot(tmp_input.s89ab, tmp_weight.s89ab);
                                result += dot(tmp_input.scdef, tmp_weight.scdef);

                }

                result += bias[n+c];
                r[c] = result;
               }
                float4 r1 = (float4)(0.0f);
                float4 r2;
                r2.x = (float)r[0];
                r2.y = (float)r[1];
                r2.z = (float)r[2];
                r2.w = (float)r[3];
                r1 = max(r1,r2);

                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
               // output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
                vstore_half4(r1,0,&output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)]);
            }
        }
    }
}
*/



//3*3 conv
__kernel void conv_3_3_kernel_half(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c) {
int x,y,z,n,i,j;

int threadId_x = get_global_id(0);
int threadId_y = get_global_id(1);
int threadId_z = get_global_id(2);

for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
half result = 0.0f;
for(y = 0 ; y < conv_h ; y++) {
int global_input_y = j * stride_h - pad_top + y;
for(x = 0 ; x < conv_w ; x++) {
int global_input_x = i * stride_w - pad_left + x;
if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
for(z = 0 ; z < conv_c ; z += 16) {
int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

half16 tmp_input = vload16(0, &input[global_input_index]);
half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

result += dot(tmp_input.s0123, tmp_weight.s0123);
result += dot(tmp_input.s4567, tmp_weight.s4567);
result += dot(tmp_input.s89ab, tmp_weight.s89ab);
result += dot(tmp_input.scdef, tmp_weight.scdef);

}
}
}
}

result += bias[n];

output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
}
}
}
}

//testopencl
__kernel void TestOpencl(
        __global const  char *input,
        __global const  char *wight,
        __global  char *output
){
    char8 result8,tem_input,tem_wight;
    char result=0;
    result8 = (char8)(0,0,0,0,0,0,0,0);
    tem_input = vload8(0,input);
    tem_wight = vload8(0,wight);

    result8= mad_sat(tem_input,tem_wight,result8);

    vstore8(result8,0,output);
}


__kernel void conv_fc_kernel_half(
        __global const half *input,
const int input_w,
const int input_h,
const int input_c,
        __global const half *conv_weight,
        __global const half *bias,
const int conv_w,
const int conv_h,
const int conv_c,
const int conv_n,
const int stride_w,
const int stride_h,
const int pad_left,
const int pad_right,
const int pad_top,
const int pad_bot,
        __global half *output,
const int output_w,
const int output_h,
const int output_c
) {
for(int threadId_x = get_global_id(0) ; threadId_x < output_c ; threadId_x += get_global_size(0)) {
int i;
int weight_start_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, threadId_x, 0, 0, 0);
float result = 0.0f;

int remaining = conv_w * conv_h * conv_c;
i = 0;
while(remaining > 0 && remaining / 16 > 0) {
half16 tmp_input = vload16(0, &input[i]);
half16 tmp_weight = vload16(0, &conv_weight[weight_start_index + i]);

result += dot(tmp_input.s0123, tmp_weight.s0123);
result += dot(tmp_input.s4567, tmp_weight.s4567);
result += dot(tmp_input.s89ab, tmp_weight.s89ab);
result += dot(tmp_input.scdef, tmp_weight.scdef);

remaining -= 16;
i += 16;
}

while(remaining > 0 && remaining / 4 > 0) {
half4 tmp_input = vload4(0, &input[i]);
half4 tmp_weight = vload4(0, &conv_weight[weight_start_index + i]);

result += dot(tmp_input, tmp_weight);

remaining -= 4;
i += 4;
}

while(remaining > 0) {
result += input[i] * conv_weight[weight_start_index + i];

remaining--;
i++;
}

result += bias[threadId_x];

output[threadId_x] = result;
//vstore_half(result, 0, &output[threadId_x]);
}
}

__kernel void activation_kernel_half(
        __global half *data,
const int activation) {
half result = data[get_global_id(0)];

switch(activation) {
case 0:
//no activation
break;
case 1:
//RAMP
result = result * (result > 0) + 0.1 * result;
break;
case 2:
//LOGISTIC
result = 1.0 / (1.0 + exp(-result));
break;
case 3:
//LEAKY
result = (result > 0) ? result : 0.1 * result;
break;
case 4:
//LINEAR
break;
case 5:
//RELU
result = (result > 0) ? result : 0.0f;
break;
}

data[get_global_id(0)] = result;
}

/*
__kernel void conv_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    int useBase3 = (input_c % 3 == 0) ? 1 : 0;

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                half result = 0.0f;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    for(x = 0 ; x < conv_w ; x++) {
                        int global_input_x = i * stride_w - pad_left + x;
                        if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                            if(useBase3 == 1) {
                                for(z = 0 ; z < conv_c ; z += 3) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);
                                    
                                    half2 tmp_input = vload2(0, &input[global_input_index]);
                                    half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                                    result += dot(tmp_input, tmp_weight);
                                    
                                    result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
                                }
                            } else {
                                for(z = 0 ; z < conv_c ; z += 16) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                                    half16 tmp_input = vload16(0, &input[global_input_index]);
                                    half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                                    result += dot(tmp_input.s0123, tmp_weight.s0123);
                                    result += dot(tmp_input.s4567, tmp_weight.s4567);
                                    result += dot(tmp_input.s89ab, tmp_weight.s89ab);
                                    result += dot(tmp_input.scdef, tmp_weight.scdef);
                                }
                            }
                        }
                    }
                }

                result += bias[n];

                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
            }
        }
    }
}
*/


__kernel void conv_kernel_float(
        __global const float *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const float *conv_weight,
        __global const float *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global float *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    int useBase3 = (input_c % 3 == 0) ? 1 : 0;

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                float result = 0.0f;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    for(x = 0 ; x < conv_w ; x++) {
                        int global_input_x = i * stride_w - pad_left + x;
                        if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                            if(useBase3 == 1) {
                                for(z = 0 ; z < conv_c ; z += 3) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                                    float2 tmp_input = vload2(0, &input[global_input_index]);
                                    float2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                                    result += dot(tmp_input, tmp_weight);

                                    result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
                                }
                            } else {
                                for(z = 0 ; z < conv_c ; z += 16) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                                    float16 tmp_input = vload16(0, &input[global_input_index]);
                                    float16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                                    result += dot(tmp_input.s0123, tmp_weight.s0123);
                                    result += dot(tmp_input.s4567, tmp_weight.s4567);
                                    result += dot(tmp_input.s89ab, tmp_weight.s89ab);
                                    result += dot(tmp_input.scdef, tmp_weight.scdef);
                                }
                            }
                        }
                    }
                }

                result += bias[n];

                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
            }
        }
    }
}

__kernel void conv_fc_kernel_float(
        __global const float *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const float *conv_weight,
        __global const float *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global float *output,
        const int output_w,
        const int output_h,
        const int output_c
) {
    for(int threadId_x = get_global_id(0) ; threadId_x < output_c ; threadId_x += get_global_size(0)) {
        int weight_start_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, threadId_x, 0, 0, 0);
        float result = 0.0f;

        int remaining = conv_w * conv_h * conv_c;
        int i = 0;
        while(remaining > 0 && remaining / 16 > 0) {
            float16 tmp_input = vload16(0, &input[i]);
            float16 tmp_weight = vload16(0, &conv_weight[weight_start_index + i]);

            result += dot(tmp_input.s0123, tmp_weight.s0123);
            result += dot(tmp_input.s4567, tmp_weight.s4567);
            result += dot(tmp_input.s89ab, tmp_weight.s89ab);
            result += dot(tmp_input.scdef, tmp_weight.scdef);

            remaining -= 16;
            i += 16;
        }

        while(remaining > 0 && remaining / 4 > 0) {
            float4 tmp_input = vload4(0, &input[i]);
            float4 tmp_weight = vload4(0, &conv_weight[weight_start_index + i]);

            result += dot(tmp_input, tmp_weight);

            remaining -= 4;
            i += 4;
        }

        while(remaining > 0) {
            result += input[i] * conv_weight[weight_start_index + i];

            remaining--;
            i++;
        }

        result += bias[threadId_x];

        output[threadId_x] = result;
    }
}

kernel void fully_connected_kernel_half(
        global const half *input_frame,
const int input_w,
const int input_h,
const int input_d,
        global const half *layer_W,
        global const half *layer_bias,
        global half *output_frame,
const int output_size){
int thrIdx = get_global_id(0);
int maxThreads = get_global_size(0);

for(int n = thrIdx; n < output_size ; n += maxThreads) {
float result = 0.0f;

int input_idx = 0;
int filter_idx = n * input_h * input_w * input_d;

int idx_remaining = input_h * input_w * input_d;

while(idx_remaining >= 4) {
half4 tmp1 = vload4(0, &input_frame[input_idx]);
half4 tmp2 = vload4(0, &layer_W[filter_idx]);
result += dot(tmp1,tmp2);

input_idx += 4;
filter_idx += 4;
idx_remaining -= 4;
}

while(idx_remaining >= 2) {
half2 tmp1 = vload2(0, &input_frame[input_idx]);
half2 tmp2 = vload2(0, &layer_W[filter_idx]);
result += dot(tmp1,tmp2);

input_idx += 2;
filter_idx += 2;
idx_remaining -= 2;
}

while(idx_remaining > 0) {
half tmp1 = input_frame[input_idx];
half tmp2 = layer_W[filter_idx];
result += tmp1 * tmp2;

idx_remaining -= 1;
}

result += layer_bias[n];

output_frame[n] = result;
}
}

__kernel void fully_connected_kernel_float(
        global const float *input_frame,
        const int input_w,
        const int input_h,
        const int input_d,
        global const float *layer_W,
        global const float *layer_bias,
        global float *output_frame,
        const int output_size
) {
    int thrIdx = get_global_id(0);
    int maxThreads = get_global_size(0);

    for(int n = thrIdx; n < output_size ; n += maxThreads) {
        float result = 0.0f;

        int input_idx = 0;
        int filter_idx = n * input_h * input_w * input_d;

        int idx_remaining = input_h * input_w * input_d;

        while(idx_remaining >= 4) {
            float4 tmp1 = vload4(0, &input_frame[input_idx]);
            float4 tmp2 = vload4(0, &layer_W[filter_idx]);
            result += dot(tmp1,tmp2);

            input_idx += 4;
            filter_idx += 4;
            idx_remaining -= 4;
        }

        while(idx_remaining > 0) {
            float tmp1 = input_frame[input_idx];
            float tmp2 = layer_W[filter_idx];
            result += tmp1 * tmp2;

            idx_remaining -= 1;
        }

        result += layer_bias[n];

        output_frame[n] = result;
    }
}

__kernel void maxpool_kernel_half(
        __global const half *input_frame,
const int input_w,
const int input_h,
const int input_d,
const int size,
const int stride_1,
const int stride_2,
const int pad_1,
const int pad_2,
const int pad_3,
const int pad_4,
        __global half *output_frame,
const int output_w,
const int output_h,
const int output_d) {

int thrId_i = get_global_id(0);
int thrId_j = get_global_id(1);
int thrId_k = get_global_id(2);

int max_i = get_global_size(0);
int max_j = get_global_size(1);
int max_k = get_global_size(2);

int i,j,k;
int x,y;

for(i = thrId_i ; i < output_w ; i += max_i) {
for(j = thrId_j ; j < output_h ; j += max_j) {
for(k = thrId_k ; k < output_d ; k += max_k) {
half max = -9999.9f;
for(x = 0 ; x < size ; x++) {
for(y = 0 ; y < size ; y++) {
int x_ = i * stride_1 + x - pad_1;
int y_ = j * stride_2 + y - pad_3;
int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
//float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : -999999.9f;
half val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : 0.0f;
max   = (val > max) ? val   : max;
}
}
output_frame[getIndexFrom3D(output_h, output_w, output_d, j, i, k)] = max;
//vstore_half(max, 0, &output_frame[getIndexFrom3D(output_h, output_w, output_d, j, i, k)]);
}
}
}
}

__kernel void maxpool_kernel_float(
        __global const float *input_frame,
        const int input_w,
        const int input_h,
        const int input_d,
        const int size,
        const int stride_1,
        const int stride_2,
        const int pad_1,
        const int pad_2,
        const int pad_3,
        const int pad_4,
        __global float *output_frame,
        const int output_w,
        const int output_h,
        const int output_c) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;

    for(i = thrId_i ; i < output_w ; i += max_i) {
        for(j = thrId_j ; j < output_h ; j += max_j) {
            for(k = thrId_k ; k < output_c ; k += max_k) {
                float max = -9999.9f;
                for(x = 0 ; x < size ; x++) {
                    for(y = 0 ; y < size ; y++) {
                        int x_ = i * stride_1 + x - pad_1;
                        int y_ = j * stride_2 + y - pad_3;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        //float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : -999999.9f;
                        float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : 0.0f;
                        max   = (val > max) ? val   : max;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, output_c, j, i, k)] = max;
            }
        }
    }
}

__kernel void cross_channels_lrn_kernel_half(
        __global const half *in, //[h x w x c]
const int channels,
const int height,
const int width,
const int k,
const int size,
const float alpha_over_size,
const float beta,
        __global half *out){

half beta_half = 0.0f;
vstore_half(beta, 0, &beta_half);

for(int w = get_global_id(0) ; w < width ; w += get_global_size(0)) {
for(int h = get_global_id(1) ; h < height ; h += get_global_size(1)) {
int offset = (h * width + w) * channels;
int head = 0;
int pre_pad = (size - 1) / 2;
int post_pad = size - pre_pad - 1;
half accum_scale = 0;

while (head < post_pad) {
half data = in[offset + head];
accum_scale += data * data;
head++;
}

while (head < size) {
half data = in[offset + head];
accum_scale += data * data;
half scale = k + accum_scale * alpha_over_size;
out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
head++;
}

while (head < channels) {
half data = in[offset + head];
accum_scale += data * data;
data = in[offset + head - size];
accum_scale -= data * data;
half scale = k + accum_scale * alpha_over_size;
out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
head++;
}

while (head < channels + post_pad) {
half data = in[offset + head - size];
accum_scale -= data * data;
half scale = k + accum_scale * alpha_over_size;
out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
head++;
}
}
}
}

__kernel void cross_channels_lrn_kernel_float(
        __global const float *input, //[h x w x c]
        const int channels,
        const int height,
        const int width,
        const int k,
        const int size,
        const float alpha_over_size,
        const float beta,
        __global float *output) {

    for(int w = get_global_id(0) ; w < width ; w += get_global_size(0)) {
        for(int h = get_global_id(1) ; h < height ; h += get_global_size(1)) {
            int offset = getIndexFrom3D(height, width, channels, h, w, 0);
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            float accum_scale = 0;

            const __global float *in = input + offset;
            __global float *out = output + offset;

            while (head < post_pad) {
                float data = in[head];
                accum_scale += data * data;
                head++;
            }

            while (head < size) {
                float data = in[head];
                accum_scale += data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels) {
                float data = in[head];
                accum_scale += data * data;
                data = in[head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels + post_pad) {
                float data = in[head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[ head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }
        }
    }
}

__kernel void activation_kernel_float(
        __global float *data,
const int activation){
float result = data[get_global_id(0)];

switch(activation) {
case 0:
//no activation
break;
case 1:
//RAMP
result = result * (result > 0) + 0.1 * result;
break;
case 2:
//LOGISTIC
result = 1.0 / (1.0 + exp(-result));
break;
case 3:
//LEAKY
result = (result > 0) ? result : 0.1 * result;
break;
case 4:
//LINEAR
break;
case 5:
//RELU
result = (result > 0) ? result : 0.0f;
break;
}

data[get_global_id(0)] = result;
}


//GPU_INT8

//无优化的第一层卷积
//first conv
//这里有符号char不能用signed char ，用char即可
__kernel void conv_first_kernel_INT8(
        __global const char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c
//    __global char *input_s,
//    __global char *weight_s
){
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);


    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                char result = 0;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    {
                        for(x = 0 ; x < conv_w ; x++) {
                            int global_input_x = i * stride_w - pad_left + x;
                            if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h)
                            {
                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, 0);
                                int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, 0);
                                char2 tmp_input = vload2(0, &input[global_input_index]);
                                char2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                                char2 result2=(char2)(0,0);
                                //这个函数计算没问题，计算结果溢出的处理方式不确定
                                result2 = mad_sat(tmp_input,tmp_weight,result2);

                                result += result2.s0 + result2.s1;
                                result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
/*                                //测试用
                                if(n==0&&j==0&&i==0){
                                    vstore2(result2,0,weight_s);
                                    vstore2(tmp_input,0,input_s);
                                    weight_s+=2;
                                    input_s+=2;
                                    weight_s[0] = result;
                                    input_s[0] = input[global_input_index + 2];
                                    weight_s++;
                                    input_s++;
                                }*/
                            }
                        }
                    }
                }

                result += bias[n];
                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
//测试用
//                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result ;
//                output[getIndexFrom3D(output_c, output_h, output_w, n, j, i)] = result > 0 ? result : 0;

            }
        }
    }
}

//无优化的DW卷积
__kernel void conv_dw_kernel_INT8(
        __global const  char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                signed char result = 0;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    for(x = 0 ; x < conv_w ; x++) {
                        int global_input_x = i * stride_w - pad_left + x;
                        if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                            int global_filter_index = getIndexFrom3D(conv_n, conv_h, conv_w, n, y, x);
                            int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, n);
                            result += input[global_input_index] * conv_weight[global_filter_index];
                        }
                    }
                }

                result += bias[n];

                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}

//1*1 点卷积
/*

__kernel void conv_1_1_kernel_INT8(
    __global const  char *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const  char *conv_weight,
    __global const  char *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global  char *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    char16 result1=(char16)(0);
    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {

                result1=(char16)(0);
                int global_input_y = j * stride_h - pad_top;
                int global_input_x = i * stride_w - pad_left;
                for(z = 0 ; z < conv_c ; z += 16) {
                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, 0, 0, z);
                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                    char16 tmp_input = vload16(0, &input[global_input_index]);
                    char16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);
                    result1=(char16)(0);
                    result1 = mad_sat(tmp_input,tmp_weight,result1);

                }
                char8 r1 = add_sat(result1.s01234567,result1.s89abcdef);
                char4 r2 = add_sat(r1.s0123,r1.s4567);
                char2 r3 = add_sat(r2.s01,r2.s23);
                char result = add_sat(r3.s0,r3.s1);

                result += bias[n];
                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}
*/

//1*1 点卷积
__kernel void conv_1_1_kernel_INT8(
        __global const  char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(1);
    int threadId_y = get_global_id(2);
    int threadId_z = get_global_id(0);
    //char result = 0;
    char16 result1=(char16)(0);

    for(j = threadId_y ; j < output_h ; j += get_global_size(2)) {
        for(i = threadId_x ; i < output_w ; i += get_global_size(1)) {
            //get_global_size(2)=output_c / 4
            for(n = threadId_z*4 ; n < output_c ; n+=get_global_size(0)*4) {
                char4 result4;
                char r[4];

                int global_input_y = j * stride_h - pad_top;
                int global_input_x = i * stride_w - pad_left;
                for (int k = 0; k < 4; ++k) {
                    result1=(char16)(0);
                    for (z = 0; z < conv_c; z += 16) {
                        int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n+k,
                                                                 0, 0, z);
                        int global_input_index = getIndexFrom3D(input_h, input_w, input_c,
                                                                global_input_y, global_input_x, z);

                        char16 tmp_input = vload16(0, &input[global_input_index]);
                        char16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                        result1 = mad_sat(tmp_input, tmp_weight, result1);

                    }
                    char8 r1 = add_sat(result1.s01234567, result1.s89abcdef);
                    char4 r2 = add_sat(r1.s0123, r1.s4567);
                    char2 r3 = add_sat(r2.s01, r2.s23);
                    r[k] = add_sat(r3.s0, r3.s1);
                    r[k] += bias[n+k];
                }


                result4 =(char4)(r[0],r[1],r[2],r[3]);

                char4 r_z = (char4)(0);
                r_z = max(result4,r_z);
                vstore4(r_z,0,&output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)]);
                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}


/*__kernel void conv_1_1_kernel_INT8(
        __global const  char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);
    //char result = 0;
    char16 result1=(char16)(0);
//    char16 result2=(char16)(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

    for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
        for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
            //get_global_size(2)=output_c / 4
            for(n = threadId_z*8 ; n < output_c ; n+=get_global_size(2)*8) {
                char8 result8;
                char r[8];
                result1=(char16)(0);
                int global_input_y = j * stride_h - pad_top;
                int global_input_x = i * stride_w - pad_left;
                for (int k = 0; k < 8; ++k) {
                    for (z = 0; z < conv_c; z += 16) {
                        int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n+k,
                                                                 0, 0, z);
                        int global_input_index = getIndexFrom3D(input_h, input_w, input_c,
                                                                global_input_y, global_input_x, z);

                        char16 tmp_input = vload16(0, &input[global_input_index]);
                        char16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);
                        result1=(char16)(0);
                        result1 = mad_sat(tmp_input, tmp_weight, result1);

                    }
                    char8 r1 = add_sat(result1.s01234567, result1.s89abcdef);
                    char4 r2 = add_sat(r1.s0123, r1.s4567);
                    char2 r3 = add_sat(r2.s01, r2.s23);
                    r[k] = add_sat(r3.s0, r3.s1);
                    r[k] += bias[n+k];
                }


                result8 =(char8)(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]);

                char8 r_z = (char8)(0);
                r_z = max(result8,r_z);
                vstore8(r_z,0,&output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)]);
                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}*/


/*__kernel void conv_1_1_kernel_INT8(
        __global const  char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);
    //char result = 0;
    char16 result1=(char16)(0);
//    char16 result2=(char16)(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

    for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
        for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
            //get_global_size(2)=output_c / 4
            for(n = threadId_z*16 ; n < output_c ; n+=get_global_size(2)*16) {
                char16 result16;
                char r[16];
                result1=(char16)(0);
                int global_input_y = j * stride_h - pad_top;
                int global_input_x = i * stride_w - pad_left;
                for (int k = 0; k < 16; ++k) {
                    for (z = 0; z < conv_c; z += 16) {
                        int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n+k,
                                                                 0, 0, z);
                        int global_input_index = getIndexFrom3D(input_h, input_w, input_c,
                                                                global_input_y, global_input_x, z);

                        char16 tmp_input = vload16(0, &input[global_input_index]);
                        char16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                        result1 = mad_sat(tmp_input, tmp_weight, result1);

                    }
                    char8 r1 = add_sat(result1.s01234567, result1.s89abcdef);
                    char4 r2 = add_sat(r1.s0123, r1.s4567);
                    char2 r3 = add_sat(r2.s01, r2.s23);
                    r[k] = add_sat(r3.s0, r3.s1);
                    r[k] += bias[n+k];
                }


                result16 =(char16)(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15]);

                char16 r_z = (char16)(0);
                r_z = max(result16,r_z);
                vstore16(r_z,0,&output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)]);
                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}*/

//3*3 conv
__kernel void conv_3_3_kernel_INT8(
        __global const  char *input,
        const int input_w,
        const int input_h,
        const int input_c,
        __global const  char *conv_weight,
        __global const  char *bias,
        const int conv_w,
        const int conv_h,
        const int conv_c,
        const int conv_n,
        const int stride_w,
        const int stride_h,
        const int pad_left,
        const int pad_right,
        const int pad_top,
        const int pad_bot,
        __global  char *output,
        const int output_w,
        const int output_h,
        const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                char result = 0;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    for(x = 0 ; x < conv_w ; x++) {
                        int global_input_x = i * stride_w - pad_left + x;
                        if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                            for(z = 0 ; z < conv_c ; z += 16) {
                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                                char16 tmp_input = vload16(0, &input[global_input_index]);
                                char16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);
                                //char16 result16;
                                char16 result16=(char16)(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                                result16 = mad_sat(tmp_input,tmp_weight,result16);
                                result += result16.s0 + result16.s1 + result16.s2 + result16.s3 + result16.s4 + result16.s5 + result16.s6 + result16.s7 + result16.s8 + result16.s9
                                          + result16.sa + result16.sb + result16.sc + result16.sd + result16.se + result16.sf;
                            }
                        }
                    }
                }

                result += bias[n];

                //output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result > 0 ? result : 0;
            }
        }
    }
}
