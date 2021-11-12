#include <layers/conv_layer.hpp>
#include <clio.hpp>
#include <basic_functions.hpp>
#include <deepsense_internal_lib.hpp>
#include <deepsense_lib.hpp>
#include <string.h>

//openmp
#include <omp.h>
//neon
#include <arm_neon.h>

//neon conv
cnn_frame *doFeedForward_CONV_1_1_NEON(cnn_frame *frame, void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int nt=((cnn_layer *)layer)->num_threads;

//    double time = 0;
//    double t0,t1;
//    t0 = get_timestamp();

    //获取卷积核
    float * mappedWeight = ((cnn_layer *)layer)->conv_layer->W;
    float * bias = ((cnn_layer *)layer)->conv_layer->bias;
    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    //将输入帧转存为cpu float帧
    frame = frame_convert_to_cpu(frame);
    float *input = frame->data;

    //创建输出帧
    cnn_frame *out_frame = frame_init(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, ((cnn_layer *)layer)->useINT8);

    int n_output = ((cnn_layer *)layer)->output_c / 6;
    int remain_n_output_start = n_output * 6;
    int in_plane_size = alignSize(frame->h*frame->w* sizeof(float),16)/ sizeof(float);
    int out_plane_size = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(float),16) / sizeof(float);
    #pragma omp parallel for num_threads(nt)
    for (int i = 0; i < ((cnn_layer *)layer)->output_c; ++i)
    {
        fill_float((out_frame->data+i*out_plane_size),((cnn_layer *)layer)->output_w*((cnn_layer *)layer)->output_h,((cnn_layer *)layer)->conv_layer->bias[i]);
    }
    float *output = out_frame->data;

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("data_pre time = %f ",time);

//    t0 = get_timestamp();

    //循环 n/6 个卷积核，n%6个卷积核未参加计算
    //每个卷积核通道上循环 C%4


    #pragma omp parallel for num_threads(nt)
    for (int m = 0 ; m < n_output; ++m)
    {
        //取输出的6个通道平面的数据，并用bias初始化
        int n = m*6;
        float * out0 = output + out_plane_size*(n+0);
        float * out1 = output + out_plane_size*(n+1);
        float * out2 = output + out_plane_size*(n+2);
        float * out3 = output + out_plane_size*(n+3);
        float * out4 = output + out_plane_size*(n+4);
        float * out5 = output + out_plane_size*(n+5);

        //bias初始化out

        int c = 0;
        for (; c+3 < ((cnn_layer *)layer)->conv_layer->c; c+=4)
        {
            float * outptr0 = out0;
            float * outptr1 = out1;
            float * outptr2 = out2;
            float * outptr3 = out3;
            float * outptr4 = out4;
            float * outptr5 = out5;

            //取输入的4个通道上的数据
            const float* img0 = input + in_plane_size*(c+0);
            const float* img1 = input + in_plane_size*(c+1);
            const float* img2 = input + in_plane_size*(c+2);
            const float* img3 = input + in_plane_size*(c+3);

            //取6个卷积核，每个卷积核中取通道数上8个数据
            const float* kernel0 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+0) + c;
            const float* kernel1 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+1) + c;
            const float* kernel2 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+2) + c;
            const float* kernel3 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+3) + c;
            const float* kernel4 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+4) + c;
            const float* kernel5 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+5) + c;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = ((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w;
            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);
            float32x4_t _k4 = vld1q_f32(kernel4);
            float32x4_t _k5 = vld1q_f32(kernel5);


            //每次取 4*h*w（C*H*W）的输入，6*4*1*1 的卷积核，得出 6*h*w的输出
            if (nn > 0)
            {
            asm volatile(
                    "pld        [%7, #128]              \n"
                    "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                    "vmla.f32   q6, q12, %e22[0]        \n"

                    "0:                                 \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

                    "vmla.f32   q7, q12, %e23[0]        \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

                    "vmla.f32   q8, q12, %e24[0]        \n"

                    "pld        [%8, #128]              \n"
                    "vld1.f32   {d26-d27}, [%8 :128]!   \n"// q13 = r1

                    "vmla.f32   q9, q12, %e25[0]        \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

                    "vmla.f32   q6, q13, %e22[1]        \n"
                    "vmla.f32   q7, q13, %e23[1]        \n"

                    "pld        [%6, #128]              \n"
                    "vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

                    "vmla.f32   q10, q12, %e26[0]       \n"
                    "vmla.f32   q11, q12, %e27[0]       \n"

                    "vmla.f32   q8, q13, %e24[1]        \n"
                    "vmla.f32   q9, q13, %e25[1]        \n"

                    "pld        [%9, #128]              \n"
                    "vld1.f32   {d28-d29}, [%9 :128]!   \n"// q14 = r2

                    "vmla.f32   q10, q13, %e26[1]       \n"
                    "vmla.f32   q11, q13, %e27[1]       \n"

                    "vmla.f32   q6, q14, %f22[0]        \n"
                    "vmla.f32   q7, q14, %f23[0]        \n"
                    "vmla.f32   q8, q14, %f24[0]        \n"
                    "vmla.f32   q9, q14, %f25[0]        \n"

                    "pld        [%10, #128]             \n"
                    "vld1.f32   {d30-d31}, [%10 :128]!  \n"// q15 = r3

                    "vmla.f32   q10, q14, %f26[0]       \n"
                    "vmla.f32   q11, q14, %f27[0]       \n"

                    "vmla.f32   q6, q15, %f22[1]        \n"
                    "vmla.f32   q7, q15, %f23[1]        \n"
                    "vmla.f32   q8, q15, %f24[1]        \n"
                    "vmla.f32   q9, q15, %f25[1]        \n"

                    "pld        [%7, #128]              \n"
                    "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                    "vmla.f32   q10, q15, %f26[1]       \n"
                    "vmla.f32   q11, q15, %f27[1]       \n"

                    "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                    "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                    "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                    "vmla.f32   q6, q12, %e22[0]        \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                    "subs       %0, #1                  \n"

                    "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                    "bne        0b                      \n"

                    "sub        %7, #16                 \n"

            : "=r"(nn),     // %0
            "=r"(outptr0),// %1
            "=r"(outptr1),// %2
            "=r"(outptr2),// %3
            "=r"(outptr3),// %4
            "=r"(outptr4),// %5
            "=r"(outptr5),// %6
            "=r"(r0),     // %7
            "=r"(r1),     // %8
            "=r"(r2),     // %9
            "=r"(r3)      // %10
            : "0"(nn),
            "1"(outptr0),
            "2"(outptr1),
            "3"(outptr2),
            "4"(outptr3),
            "5"(outptr4),
            "6"(outptr5),
            "7"(r0),
            "8"(r1),
            "9"(r2),
            "10"(r3),
            "w"(_k0),     // %22
            "w"(_k1),     // %23
            "w"(_k2),     // %24
            "w"(_k3),     // %25
            "w"(_k4),     // %26
            "w"(_k5)      // %27
            : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
            }

            //(w*h)%4 的部分计算
            for(;remain>0;--remain)
            {
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3];
                float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }

        //C%4 的部分计算
        for (; c < ((cnn_layer *)layer)->conv_layer->c; c++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;

            const float* img0 = input + in_plane_size*c;  //这个地方

            const float* kernel0 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+0) + c;
            const float* kernel1 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+1) + c;
            const float* kernel2 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+2) + c;
            const float* kernel3 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+3) + c;
            const float* kernel4 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+4) + c;
            const float* kernel5 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(n+5) + c;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];
            const float k4 = kernel4[0];
            const float k5 = kernel5[0];

            const float* r0 = img0;

            int size = ((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w;

            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
            float32x4_t _k4 = vdupq_n_f32(k4);
            float32x4_t _k5 = vdupq_n_f32(k5);

            if (nn > 0)
            {
                asm volatile(
                "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "0:                                 \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                        "vmla.f32   q6, q12, %q16           \n"// outptr0 += r0*_k0

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

                        "vmla.f32   q7, q12, %q17           \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

                        "vmla.f32   q8, q12, %q18           \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

                        "vmla.f32   q9, q12, %q19           \n"

                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

                        "vmla.f32   q10, q12, %q20          \n"
                        "vmla.f32   q11, q12, %q21          \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                        "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                        "bne        0b                      \n"

                        "sub        %7, #16                 \n"

                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(outptr4),// %5
                "=r"(outptr5),// %6
                "=r"(r0)      // %7
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(r0),
                "w"(_k0),     // %16
                "w"(_k1),     // %17
                "w"(_k2),     // %18
                "w"(_k3),     // %19
                "w"(_k4),     // %20
                "w"(_k5)      // %21
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
                );
            }
            for (; remain>0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;
                float sum4 = *r0 * k4;
                float sum5 = *r0 * k5;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }
    }

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("1 part time = %f ",time);

//    t0 = get_timestamp();

    n_output = (((cnn_layer *)layer)->output_c - remain_n_output_start) >> 2;

    #pragma omp parallel for num_threads(nt)
    for (int pp=0; pp<n_output; pp++) {
        int p = remain_n_output_start + pp * 4;

        float *out0 = output + out_plane_size * (p + 0);
        float *out1 = output + out_plane_size * (p + 1);
        float *out2 = output + out_plane_size * (p + 2);
        float *out3 = output + out_plane_size * (p + 3);

        //bias初始化out

        int q = 0;
        for (; q + 3 < ((cnn_layer *)layer)->conv_layer->c; q += 4) {
            float *outptr0 = out0;
            float *outptr1 = out1;
            float *outptr2 = out2;
            float *outptr3 = out3;

            const float *img0 = input + in_plane_size * (q + 0);
            const float *img1 = input + in_plane_size * (q + 1);
            const float *img2 = input + in_plane_size * (q + 2);
            const float *img3 = input + in_plane_size * (q + 3);

            const float *kernel0 = mappedWeight + (p + 0) * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel1 = mappedWeight + (p + 1) * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel2 = mappedWeight + (p + 2) * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel3 = mappedWeight + (p + 3) * ((cnn_layer *)layer)->conv_layer->c + q;

            const float *r0 = img0;
            const float *r1 = img1;
            const float *r2 = img2;
            const float *r3 = img3;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);

            if (nn > 0) {
                asm volatile(
                "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "0:                                 \n"

                        "vmla.f32   q8, q6, %e18[0]         \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q9, q7, %e18[0]         \n"

                        "vmla.f32   q10, q6, %e19[0]        \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q11, q7, %e19[0]        \n"

                        "vmla.f32   q12, q6, %e20[0]        \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q13, q7, %e20[0]        \n"

                        "pld        [%6, #256]              \n"
                        "vld1.f32   {d8-d11}, [%6 :128]!    \n"

                        "vmla.f32   q14, q6, %e21[0]        \n"
                        "vmla.f32   q15, q7, %e21[0]        \n"

                        "vmla.f32   q8, q4, %e18[1]         \n"
                        "vmla.f32   q9, q5, %e18[1]         \n"

                        "vmla.f32   q10, q4, %e19[1]        \n"
                        "vmla.f32   q11, q5, %e19[1]        \n"

                        "vmla.f32   q12, q4, %e20[1]        \n"
                        "vmla.f32   q13, q5, %e20[1]        \n"

                        "pld        [%7, #256]              \n"
                        "vld1.f32   {d12-d15}, [%7 :128]!   \n"

                        "vmla.f32   q14, q4, %e21[1]        \n"
                        "vmla.f32   q15, q5, %e21[1]        \n"

                        "vmla.f32   q8, q6, %f18[0]         \n"
                        "vmla.f32   q9, q7, %f18[0]         \n"

                        "vmla.f32   q10, q6, %f19[0]        \n"
                        "vmla.f32   q11, q7, %f19[0]        \n"

                        "vmla.f32   q12, q6, %f20[0]        \n"
                        "vmla.f32   q13, q7, %f20[0]        \n"

                        "pld        [%8, #256]              \n"
                        "vld1.f32   {d8-d11}, [%8 :128]!    \n"

                        "vmla.f32   q14, q6, %f21[0]        \n"
                        "vmla.f32   q15, q7, %f21[0]        \n"

                        "vmla.f32   q8, q4, %f18[1]         \n"
                        "vmla.f32   q9, q5, %f18[1]         \n"

                        "vmla.f32   q10, q4, %f19[1]        \n"
                        "vmla.f32   q11, q5, %f19[1]        \n"

                        "vmla.f32   q12, q4, %f20[1]        \n"
                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "vmla.f32   q13, q5, %f20[1]        \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "vmla.f32   q14, q4, %f21[1]        \n"
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"

                        "vmla.f32   q15, q5, %f21[1]        \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"

                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"

                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(r0),     // %5
                "=r"(r1),     // %6
                "=r"(r2),     // %7
                "=r"(r3)      // %8
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "6"(r1),
                "7"(r2),
                "8"(r3),
                "w"(_k0),     // %18
                "w"(_k1),     // %19
                "w"(_k2),     // %20
                "w"(_k3)      // %21
                : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain > 0; remain--) {
                // TODO neon optimize
                float sum0 =
                        *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 =
                        *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 =
                        *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 =
                        *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
        for (; q<((cnn_layer *)layer)->conv_layer->c; q++) {
            float *outptr0 = out0;
            float *outptr1 = out1;
            float *outptr2 = out2;
            float *outptr3 = out3;

            const float *img0 = input + in_plane_size * (q + 0);

            const float *kernel0 = mappedWeight + p * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel1 = mappedWeight + (p + 1) * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel2 = mappedWeight + (p + 2) * ((cnn_layer *)layer)->conv_layer->c + q;
            const float *kernel3 = mappedWeight + (p + 3) * ((cnn_layer *)layer)->conv_layer->c + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float *r0 = img0;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);

            if (nn > 0) {
                asm volatile(
                "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "0:                                 \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "vmla.f32   q8, q6, %q12            \n"
                        "vmla.f32   q9, q7, %q12            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q10, q6, %q13           \n"
                        "vmla.f32   q11, q7, %q13           \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q12, q6, %q14           \n"
                        "vmla.f32   q13, q7, %q14           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q14, q6, %q15           \n"
                        "vmla.f32   q15, q7, %q15           \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"
                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(r0)      // %5
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "w"(_k0),     // %12
                "w"(_k1),     // %13
                "w"(_k2),     // %14
                "w"(_k3)      // %15
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain > 0; remain--) {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("2 part time = %f ",time);

//    t0 = get_timestamp();

    remain_n_output_start += n_output << 2;

    #pragma omp parallel for num_threads(nt)
    for (int p=remain_n_output_start; p<((cnn_layer *)layer)->output_c; p++)
    {
        float * out = output + out_plane_size * p;
        //bias初始化

        int q = 0;

        for (; q+3<((cnn_layer *)layer)->conv_layer->c; q+=4) {
            float *outptr = out;

            const float *img0 = input + in_plane_size * (q + 0);
            const float *img1 = input + in_plane_size * (q + 1);
            const float *img2 = input + in_plane_size * (q + 2);
            const float *img3 = input + in_plane_size * (q + 3);

            const float *kernel0 = mappedWeight + p * ((cnn_layer *)layer)->conv_layer->c + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float *r0 = img0;
            const float *r1 = img1;
            const float *r2 = img2;
            const float *r3 = img3;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);

            if (nn > 0) {
                asm volatile(
                "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q12        \n"
                        "vmla.f32   q1, q3, %q12        \n"
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d4-d7}, [%3 :128]! \n"
                        "vmla.f32   q0, q2, %q13        \n"
                        "vmla.f32   q1, q3, %q13        \n"
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d4-d7}, [%4 :128]! \n"
                        "vmla.f32   q0, q2, %q14        \n"
                        "vmla.f32   q1, q3, %q14        \n"
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"
                        "vmla.f32   q0, q2, %q15        \n"
                        "vmla.f32   q1, q3, %q15        \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                "=r"(outptr), // %1
                "=r"(r0),     // %2
                "=r"(r1),     // %3
                "=r"(r2),     // %4
                "=r"(r3)      // %5
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "5"(r3),
                "w"(_k0),     // %12
                "w"(_k1),     // %13
                "w"(_k2),     // %14
                "w"(_k3)      // %15
                : "cc", "memory", "q0", "q1", "q2", "q3"
                );
            }
            for (; remain > 0; remain--) {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }
        }
        for (; q<((cnn_layer *)layer)->conv_layer->c; q++) {
            float *outptr = out;

            const float *img0 = input + in_plane_size * (q + 0);

            const float *kernel0 = mappedWeight + p * ((cnn_layer *)layer)->conv_layer->c + q;
            const float k0 = kernel0[0];

            const float *r0 = img0;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);

            if (nn > 0) {
                asm volatile(
                "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q6         \n"
                        "vmla.f32   q1, q3, %q6         \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                "=r"(outptr), // %1
                "=r"(r0)      // %2
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "w"(_k0)      // %6
                : "cc", "memory", "q0", "q1", "q2", "q3"
                );
            }
            for (; remain > 0; remain--) {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
        }
    }

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("3 part time = %f ",time);

    frame_free(frame);

    doFeedForward_Activation(out_frame, ((cnn_layer *)layer)->activation);

    return out_frame;
}

cnn_frame *doFeedForward_CONV_DW_S1_NEON(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int nt = ((cnn_layer *)layer)->num_threads;

//    double time = 0;
//    double t0,t1;
//    t0 = get_timestamp();

    //获取卷积核
    float * mappedWeight = ((cnn_layer *)layer)->conv_layer->W;
    const float* bias = ((cnn_layer *)layer)->conv_layer->bias;

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    //将输入帧转存为cpu float帧
    frame = frame_convert_to_cpu(frame);
    frame = frame_cpu_pad(frame);
    float *input = frame->data;

    //创建输出帧
    cnn_frame *out_frame = frame_init(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, ((cnn_layer *)layer)->useINT8);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(float),16) / sizeof(float);
//    int weightstep = ((cnn_layer *)layer)->conv_layer->h*((cnn_layer *)layer)->conv_layer->w;
    int inputstep = alignSize(frame->h*frame->w* sizeof(float),16)/ sizeof(float);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int w = frame->w;
    const int group = frame->c;
#pragma omp parallel for num_threads(nt)
    for (int i = 0; i < ((cnn_layer *)layer)->output_c; ++i) {
        fill_float((out_frame->data+i*outputstep),outw*outh,((cnn_layer *)layer)->conv_layer->bias[i]);
    }
    float *output = out_frame->data;

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("data_pre time = %f ",time);

//    t0 = get_timestamp();

    #pragma omp parallel for num_threads(nt)
    for (int g=0; g<group; g++)
    {
        float * outptr = output + g*outputstep;
        float * outptr2 = outptr + outw;

        const float* img0 = input + g*inputstep;
        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;
        const float* r3 = img0 + w*3;

        const float bias0 = bias[g];
        float32x4_t _bias0 = vdupq_n_f32(bias0);

        const float* kernel0 = mappedWeight + g*9;
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0+3);
        float32x4_t _k678x = vld1q_f32(kernel0+6);
        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        int i = 0;
        for (; i+1 < outh; i+=2)
        {
            int nn = outw >> 2;
            int remain = outw & 3;

            if (nn > 0)
            {
                asm volatile(
                "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n"// r0
                        "add        %3, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "0:                             \n"

                        "vmul.f32   q7, q9, %e14[0]     \n"

                        "vand       q13, %q17, %q17     \n"// q13 = _bias0
                        "vmul.f32   q6, q11, %e14[1]    \n"
                        "vmla.f32   q13, q12, %f14[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n"// r1
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e15[1]    \n"
                        "vmla.f32   q13, q12, %f15[0]   \n"

                        "vmul.f32   q8, q9, %e14[0]     \n"

                        "vand       q15, %q17, %q17     \n"// q15 = _bias0
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        "vmla.f32   q15, q12, %f14[0]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5 :64] \n"// r2
                        "add        %5, #16             \n"

                        "vmla.f32   q7, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e16[1]    \n"
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        "vmla.f32   q8, q9, %e15[0]     \n"
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n"// r3
                        "add        %6, #16             \n"

                        "vmla.f32   q8, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q14, q11, %e16[1]   \n"
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n"// r0

                        "vadd.f32   q8, q8, q14         \n"
                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q8, q8, q15         \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "add        %3, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"
                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %3, #16             \n"
                : "=r"(nn),         // %0
                "=r"(outptr),     // %1
                "=r"(outptr2),    // %2
                "=r"(r0),         // %3
                "=r"(r1),         // %4
                "=r"(r2),         // %5
                "=r"(r3)          // %6
                : "0"(nn),
                "1"(outptr),
                "2"(outptr2),
                "3"(r0),
                "4"(r1),
                "5"(r2),
                "6"(r3),
                "w"(_k012x),      // %14
                "w"(_k345x),      // %15
                "w"(_k678x),      // %16
                "w"(_bias0)       // %17
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain>0; remain--)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r30 = vld1q_f32(r3);

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                float32x4_t _sum2 = vmulq_f32(_r10, _k012x);
                _sum2 = vmlaq_f32(_sum2, _r20, _k345x);
                _sum2 = vmlaq_f32(_sum2, _r30, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);
                _sum2 = vsetq_lane_f32(bias0, _sum2, 3);

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

                float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

                *outptr = vget_lane_f32(_sss2, 0);
                *outptr2 = vget_lane_f32(_sss2, 1);

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
                outptr2++;
            }
            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr += outw;
            outptr2 += outw;
        }

        for (; i < outh; i++)
        {
            int nn = outw >> 2;
            int remain = outw & 3;

            if (nn > 0)
            {
                asm volatile(
                "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n"// r0
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "0:                             \n"

                        "vmul.f32   q7, q8, %e10[0]     \n"

                        "vand       q14, %q13, %q13     \n"// q14 = _bias0
                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmla.f32   q14, q11, %f10[0]   \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n"// r1
                        "add        %3, #16             \n"

                        "vmla.f32   q7, q8, %e11[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n"// r2
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n"// r0
                        "add        %2, #16             \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                : "=r"(nn),         // %0
                "=r"(outptr),     // %1
                "=r"(r0),         // %2
                "=r"(r1),         // %3
                "=r"(r2)          // %4
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "w"(_k012x),      // %10
                "w"(_k345x),      // %11
                "w"(_k678x),      // %12
                "w"(_bias0)       // %13
                : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain>0; remain--)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r20 = vld1q_f32(r2);

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                *outptr = vget_lane_f32(_ss, 0);

                r0++;
                r1++;
                r2++;
                outptr++;
            }
            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }

    frame_free(frame);

    doFeedForward_Activation(out_frame, ((cnn_layer *)layer)->activation);

    return out_frame;
}

cnn_frame *doFeedForward_CONV_DW_S2_NEON(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int nt = ((cnn_layer *)layer)->num_threads;

//    double time = 0;
//    double t0,t1;
//    t0 = get_timestamp();

    //获取卷积核
    float * mappedWeight = ((cnn_layer *)layer)->conv_layer->W;
    const float* bias = ((cnn_layer *)layer)->conv_layer->bias;

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    //将输入帧转存为cpu float帧
    frame = frame_convert_to_cpu(frame);
    frame = frame_cpu_pad(frame);
    float *input = frame->data;

    //创建输出帧
    cnn_frame *out_frame = frame_init(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, ((cnn_layer *)layer)->useINT8);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(float),16) / sizeof(float);
//    int weightstep = ((cnn_layer *)layer)->conv_layer->h*((cnn_layer *)layer)->conv_layer->w;
    int inputstep = alignSize(frame->h*frame->w* sizeof(float),16)/ sizeof(float);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int w = frame->w;
    const int group = frame->c;
    const int tailstep = w - 2*outw + w;
    #pragma omp parallel for num_threads(nt)
    for (int i = 0; i < ((cnn_layer *)layer)->output_c; ++i)
    {
        fill_float((out_frame->data+i*outputstep),outw*outh,((cnn_layer *)layer)->conv_layer->bias[i]);
    }
    float *output = out_frame->data;

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("data_pre time = %f ",time);

//    t0 = get_timestamp();

    #pragma omp parallel for num_threads(nt)
    for (int g=0; g<group; g++) {
        float *outptr = output + g * outputstep;

        const float *img0 = input + g * inputstep;
        const float *r0 = img0;
        const float *r1 = img0 + w;
        const float *r2 = img0 + w * 2;

        const float bias0 = bias[g];
        float32x4_t _bias0 = vdupq_n_f32(bias0);

        const float *kernel0 = mappedWeight + g * 9;
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0 + 3);
        float32x4_t _k678x = vld1q_f32(kernel0 + 6);
        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        int i = 0;
        for (; i < outh; i++) {             //
            int nn = outw >> 2;
            int remain = outw & 3;

            if (nn > 0) {
                asm volatile(
                "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"   //r0

                        "vand       q11, %q13, %q13     \n"   //bias

                        "0:                             \n"
                        "vmul.f32   q0, q2, %e10[0]     \n"   //k012
                        "vmul.f32   q10, q3, %e10[1]    \n"

                        "pld        [%2, #128]          \n"
                        "vld2.f32   {d16-d17}, [%2]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f10[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"

                        "vmla.f32   q0, q2, %e11[0]     \n"
                        "vmla.f32   q10, q3, %e11[1]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d16-d17}, [%3]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f11[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"

                        "vmla.f32   q0, q2, %e12[0]     \n"
                        "vmla.f32   q10, q3, %e12[1]    \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d16-d17}, [%4]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f12[0]    \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vadd.f32   q0, q0, q10         \n"
                        "vadd.f32   q0, q0, q11         \n"

                        "vand       q11, %q13, %q13     \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                "=r"(outptr), // %1
                "=r"(r0),     // %2
                "=r"(r1),     // %3
                "=r"(r2)      // %4
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "w"(_k012x),  // %10
                "w"(_k345x),  // %11
                "w"(_k678x),  // %12
                "w"(_bias0)   // %13
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain > 0; remain--) {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r20 = vld1q_f32(r2);

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                *outptr = vget_lane_f32(_ss, 0);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("dw s2 conv time = %f ",time);

    frame_free(frame);

    doFeedForward_Activation(out_frame, ((cnn_layer *)layer)->activation);

    return out_frame;
}

cnn_frame *doFeedForward_CONV_3_3_S2_NEON(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int nt = ((cnn_layer *)layer)->num_threads;

//    double time = 0;
//    double t0,t1;
//    t0 = get_timestamp();

    //获取卷积核
    float * kernel = ((cnn_layer *)layer)->conv_layer->W;
    const float* bias = ((cnn_layer *)layer)->conv_layer->bias;

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    //将输入帧转存为cpu float帧
    frame->useINT8=0;
    frame = frame_convert_to_cpu(frame);
    frame = frame_cpu_pad(frame);
    float *input = frame->data;

    //创建输出帧
    ((cnn_layer *)layer)->useINT8=0;
    cnn_frame *out_frame = frame_init(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, ((cnn_layer *)layer)->useINT8);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(float),16) / sizeof(float);
    int weightstep = ((cnn_layer *)layer)->conv_layer->h*((cnn_layer *)layer)->conv_layer->w*((cnn_layer *)layer)->conv_layer->c;
    int inputstep = alignSize(frame->h*frame->w* sizeof(float),16)/ sizeof(float);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int outch = out_frame->c;
    int w = frame->w;
    int inch = frame->c;
    const int tailstep = w - 2*outw + w;
#pragma omp parallel for num_threads(nt)
    for (int i = 0; i < ((cnn_layer *)layer)->output_c; ++i)
    {
        fill_float((out_frame->data+i*outputstep),outw*outh,((cnn_layer *)layer)->conv_layer->bias[i]);
    }
    float *output = out_frame->data;

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("data_pre time = %f ",time);

//    t0 = get_timestamp();

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;


    #pragma omp parallel for num_threads(nt)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        float* out0 = out_frame->data + p*outputstep;
        float* out1 = out_frame->data + (p+1)*outputstep;


        const float* k0 = kernel + p*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const float* img0 = frame->data + q*inputstep;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0+3);
            float32x4_t _k06 = vld1q_f32(k0+6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1+3);
            float32x4_t _k16 = vld1q_f32(k1+6);

            int i = 0;
            for (; i < outh; i++)
            {
                int nn = outw >> 2;
                int remain = outw & 3;

                if (nn > 0)
                {
                    asm volatile(
                    "pld        [%3, #256]          \n"
                            "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                            "0:                             \n"

                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d12-d13}, [%1]     \n"// q6 = _sum0

                            "vmul.f32   q12, q8, %e12[0]    \n"

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d14-d15}, [%2]     \n"// q7 = _sum1

                            "vmul.f32   q13, q8, %e15[0]    \n"

                            "pld        [%3, #128]          \n"
                            "vld2.f32   {d20-d21}, [%3]     \n"// q10

                            "vmla.f32   q6, q9, %e12[1]     \n"

                            "vext.32    q11, q8, q10, #1    \n"

                            "vmla.f32   q7, q9, %e15[1]     \n"

                            "pld        [%4, #256]          \n"
                            "vld2.f32   {d16-d19}, [%4]!    \n"// r1

                            "vmla.f32   q12, q11, %f12[0]   \n"
                            "vmla.f32   q13, q11, %f15[0]   \n"

                            "pld        [%4, #128]          \n"
                            "vld2.f32   {d20-d21}, [%4]     \n"

                            "vmla.f32   q6, q8, %e13[0]     \n"
                            "vmla.f32   q7, q8, %e16[0]     \n"

                            "vext.32    q11, q8, q10, #1    \n"

                            "vmla.f32   q12, q9, %e13[1]    \n"
                            "vmla.f32   q13, q9, %e16[1]    \n"

                            "pld        [%5, #256]          \n"
                            "vld2.f32   {d16-d19}, [%5]!    \n"// r2

                            "vmla.f32   q6, q11, %f13[0]    \n"
                            "vmla.f32   q7, q11, %f16[0]    \n"

                            "pld        [%5, #128]          \n"
                            "vld2.f32   {d20-d21}, [%5]     \n"

                            "vmla.f32   q12, q8, %e14[0]    \n"
                            "vmla.f32   q13, q8, %e17[0]    \n"

                            "vext.32    q11, q8, q10, #1    \n"

                            "vmla.f32   q6, q9, %e14[1]     \n"
                            "vmla.f32   q7, q9, %e17[1]     \n"

                            "vmla.f32   q12, q11, %f14[0]   \n"
                            "vmla.f32   q13, q11, %f17[0]   \n"

                            "pld        [%3, #256]          \n"
                            "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                            "vadd.f32   q6, q6, q12         \n"
                            "vadd.f32   q7, q7, q13         \n"

                            "subs       %0, #1              \n"

                            "vst1.f32   {d12-d13}, [%1]!    \n"
                            "vst1.f32   {d14-d15}, [%2]!    \n"

                            "bne        0b                  \n"
                            "sub        %3, #32             \n"

                    : "=r"(nn),         // %0
                    "=r"(outptr0),    // %1
                    "=r"(outptr1),    // %2
                    "=r"(r0),         // %3
                    "=r"(r1),         // %4
                    "=r"(r2)          // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2),
                    "w"(_k00),      // %12
                    "w"(_k03),      // %13
                    "w"(_k06),      // %14
                    "w"(_k10),      // %15
                    "w"(_k13),      // %16
                    "w"(_k16)       // %17
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
                for (; remain>0; remain--)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);

                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9;
            k1 += 9;

        }
    }

    #pragma omp parallel for num_threads(nt)
    for (int p=remain_outch_start; p<outch; p++)
    {
        float* out = out_frame->data + p*outputstep;

        const float* kernel0 = kernel + p*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = frame->data + q*inputstep;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);

            int i = 0;
            for (; i < outh; i++)
            {
                int nn = outw >> 2;
                int remain = outw & 3;

                if (nn > 0)
                {
                    asm volatile(
                    "pld        [%2, #256]          \n"
                            "vld2.f32   {d4-d7}, [%2]!      \n"

                            "0:                             \n"
                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d0-d1}, [%1]       \n"

                            "vmla.f32   q0, q2, %e10[0]     \n"
                            "vmul.f32   q10, q3, %e10[1]    \n"

                            "pld        [%2, #128]          \n"
                            "vld2.f32   {d16-d17}, [%2]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmul.f32   q11, q1, %f10[0]    \n"

                            "pld        [%3, #256]          \n"
                            "vld2.f32   {d4-d7}, [%3]!      \n"

                            "vmla.f32   q0, q2, %e11[0]     \n"
                            "vmla.f32   q10, q3, %e11[1]    \n"

                            "pld        [%3, #128]          \n"
                            "vld2.f32   {d16-d17}, [%3]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmla.f32   q11, q1, %f11[0]    \n"

                            "pld        [%4, #256]          \n"
                            "vld2.f32   {d4-d7}, [%4]!      \n"

                            "vmla.f32   q0, q2, %e12[0]     \n"
                            "vmla.f32   q10, q3, %e12[1]    \n"

                            "pld        [%4, #128]          \n"
                            "vld2.f32   {d16-d17}, [%4]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmla.f32   q11, q1, %f12[0]    \n"

                            "pld        [%2, #256]          \n"
                            "vld2.f32   {d4-d7}, [%2]!      \n"

                            "vadd.f32   q0, q0, q10         \n"
                            "vadd.f32   q0, q0, q11         \n"

                            "subs       %0, #1              \n"
                            "vst1.f32   {d0-d1}, [%1]!      \n"
                            "bne        0b                  \n"
                            "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                    "=r"(outptr), // %1
                    "=r"(r0),     // %2
                    "=r"(r1),     // %3
                    "=r"(r2)      // %4
                    : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "w"(_k0123),  // %10
                    "w"(_k3456),  // %11
                    "w"(_k6789)   // %12
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
            kernel0 += 9;
        }
    }

//    t1 = get_timestamp();
//    time = (t1 - t0) / 1000.0L;
//    LOGI("3*3 s2 conv time = %f ",time);

    frame_free(frame);

    doFeedForward_Activation(out_frame, ((cnn_layer *)layer)->activation);

    return out_frame;
}

//neon int8
cnn_frame *doFeedForward_CONV_1_1_NEON_S8(cnn_frame *frame, void *layer){
    LOGI("Running function %s", __PRETTY_FUNCTION__);
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d k=%d k=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * mappedWeight = ((cnn_layer *)layer)->conv_layer->W_S8;

    //将输入帧转存为cpu int8帧
    frame = frame_convert_to_cpu_int8(frame);
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int nn_outch = ((cnn_layer *)layer)->output_c / 4;
    int remain_n_output_start = nn_outch * 4;
    int in_plane_size = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);
    int out_plane_size = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);
    int inch=((cnn_layer *)layer)->conv_layer->c;
    int outw=((cnn_layer *)layer)->output_w;
    int outh=((cnn_layer *)layer)->output_h;
    int outch=((cnn_layer *)layer)->output_c;

    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        int * out0 =  (output + out_plane_size*(p+0));
        int * out1 =  (output + out_plane_size*(p+1));
        int * out2 =  (output + out_plane_size*(p+2));
        int * out3 =  (output + out_plane_size*(p+3));


        fill(out0,outh*outw,0);
        fill(out1,outh*outw,0);
        fill(out2,outh*outw,0);
        fill(out3,outh*outw,0);
        //bias初始化out

        int q = 0;

        for (; q+7<((cnn_layer *)layer)->conv_layer->c; q+=8)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* r0 = input + in_plane_size*(q+0);
            const signed char* r1 = input + in_plane_size*(q+1);
            const signed char* r2 = input + in_plane_size*(q+2);
            const signed char* r3 = input + in_plane_size*(q+3);
            const signed char* r4 = input + in_plane_size*(q+4);
            const signed char* r5 = input + in_plane_size*(q+5);
            const signed char* r6 = input + in_plane_size*(q+6);
            const signed char* r7 = input + in_plane_size*(q+7);

            const signed char* kernel0 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+0) + q;
            const signed char* kernel1 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+1) + q;
            const signed char* kernel2 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+2) + q;
            const signed char* kernel3 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+3) + q;

            int size = outh*outw;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                asm volatile(
                "vld1.s8    d18, [%0]   \n"
                        "vld1.s8    d19, [%1]   \n"
                        "vld1.s8    d24, [%2]   \n"
                        "vld1.s8    d25, [%3]   \n"
                : "=r"(kernel0), // %0   64bit 8个s8
                "=r"(kernel1), // %1
                "=r"(kernel2), // %2
                "=r"(kernel3)  // %3
                : "0"(kernel0),
                "1"(kernel1),
                "2"(kernel2),
                "3"(kernel3)
                :
                );

                asm volatile(
                "0:                            \n"
                        //ld r0-r7
                        "pld        [%5, #64]          \n"
                        "vld1.s8    {d0}, [%5 :64]!    \n" //r0

                        "pld        [%6, #64]          \n"
                        "vld1.s8    {d1}, [%6 :64]!    \n" //r1

                        "pld        [%7, #64]          \n"
                        "vld1.s8    {d2}, [%7 :64]!    \n" //r2

                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d3}, [%8 :64]!    \n" //r3

                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d4}, [%9 :64]!    \n" //r4

                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d5}, [%10 :64]!   \n" //r5

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6}, [%11 :64]!   \n" //r6

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d7}, [%12 :64]!   \n" //r7
                        //###########################################
                        //load inch kernel_0 k0-k7
                        "vdup.s8    d8, d18[0]          \n"
                        "vdup.s8    d9, d18[1]          \n"
                        "vdup.s8    d10, d18[2]         \n"
                        "vdup.s8    d11, d18[3]         \n"
                        "vdup.s8    d12, d18[4]         \n"
                        "vdup.s8    d13, d18[5]         \n"
                        "vdup.s8    d14, d18[6]         \n"
                        "vdup.s8    d15, d18[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d20-d23}, [%1:128] \n" //outptr0_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%1:128]!\n"
                        //###########################################
                        //load inch kernel_1 k0-k7
                        "vdup.s8    d8, d19[0]          \n"
                        "vdup.s8    d9, d19[1]          \n"
                        "vdup.s8    d10, d19[2]         \n"
                        "vdup.s8    d11, d19[3]         \n"
                        "vdup.s8    d12, d19[4]         \n"
                        "vdup.s8    d13, d19[5]         \n"
                        "vdup.s8    d14, d19[6]         \n"
                        "vdup.s8    d15, d19[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr1_s32
                        "pld        [%2, #256]          \n"
                        "vld1.32    {d20-d23}, [%2:128] \n" //outptr1_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%2:128]!\n"
                        //############################################
                        //load inch kernel_2 k0-k7
                        "vdup.s8    d8, d24[0]          \n"
                        "vdup.s8    d9, d24[1]          \n"
                        "vdup.s8    d10, d24[2]         \n"
                        "vdup.s8    d11, d24[3]         \n"
                        "vdup.s8    d12, d24[4]         \n"
                        "vdup.s8    d13, d24[5]         \n"
                        "vdup.s8    d14, d24[6]         \n"
                        "vdup.s8    d15, d24[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr2_s32
                        "pld        [%3, #256]          \n"
                        "vld1.32    {d20-d23}, [%3:128] \n" //outptr2_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%3:128]!\n"
                        //#############################################
                        //load inch kernel_3 k0-k7
                        "vdup.s8    d8, d25[0]          \n"
                        "vdup.s8    d9, d25[1]          \n"
                        "vdup.s8    d10, d25[2]         \n"
                        "vdup.s8    d11, d25[3]         \n"
                        "vdup.s8    d12, d25[4]         \n"
                        "vdup.s8    d13, d25[5]         \n"
                        "vdup.s8    d14, d25[6]         \n"
                        "vdup.s8    d15, d25[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr3_s32
                        "pld        [%4, #256]          \n"
                        "vld1.32    {d20-d23}, [%4:128] \n" //outptr3_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%4:128]!\n"

                        //next
                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),          // %0
                "=r"(outptr0),     // %1
                "=r"(outptr1),     // %2
                "=r"(outptr2),     // %3
                "=r"(outptr3),     // %4
                "=r"(r0),          // %5
                "=r"(r1),          // %6
                "=r"(r2),          // %7
                "=r"(r3),          // %8
                "=r"(r4),          // %9
                "=r"(r5),          // %10
                "=r"(r6),          // %11
                "=r"(r7)           // %12
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "6"(r1),
                "7"(r2),
                "8"(r3),
                "9"(r4),
                "10"(r5),
                "11"(r6),
                "12"(r7)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11", "q13", "q14", "q15"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                int sum1 = (int)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                int sum2 = (int)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                int sum3 = (int)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* img0_s8 = input + in_plane_size*q;

            const signed char* kernel0 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+0) + q;
            const signed char* kernel1 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+1) + q;
            const signed char* kernel2 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+2) + q;
            const signed char* kernel3 = mappedWeight + ((cnn_layer *)layer)->conv_layer->c*(p+3) + q;

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];

            const signed char* r0 = img0_s8;

            int size = outh*outw;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);

            if (nn > 0)
            {
                asm volatile(
                "0:                             \n"
                        //load r0
                        "pld        [%5, #64]           \n"
                        "vld1.s8    {d8}, [%5 :64]!     \n"

                        //mla
                        "vmull.s8   q5, d8, %12         \n"
                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d12-d15}, [%1]     \n"
                        "vmovl.s16  q8, d10             \n"
                        "vmovl.s16  q9, d11             \n"
                        "vadd.s32   q6, q8              \n"
                        "vadd.s32   q7, q9              \n"
                        "vst1.32    {d12-d15}, [%1]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %13         \n"
                        //outptr1_s32
                        "pld        [%2, #256]          \n"
                        "vld1.32    {d12-d15}, [%2]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%2]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %14         \n"
                        //outptr0_s32
                        "pld        [%3, #256]          \n"
                        "vld1.32    {d12-d15}, [%3]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%3]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %15         \n"
                        //outptr0_s32
                        "pld        [%4, #256]          \n"
                        "vld1.32    {d12-d15}, [%4]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),             // %0
                "=r"(outptr0),        // %1
                "=r"(outptr1),        // %2
                "=r"(outptr2),        // %3
                "=r"(outptr3),        // %4
                "=r"(r0)              // %5
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "w"(_k0),             // %12
                "w"(_k1),             // %13
                "w"(_k2),             // %14
                "w"(_k3)              // %15
                : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * k0;
                int sum1 = (int)*r0 * k1;
                int sum2 = (int)*r0 * k2;
                int sum3 = (int)*r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("2 part time = %f ",time);

    t0 = get_timestamp();

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int p=remain_n_output_start; p<((cnn_layer *)layer)->output_c; p++)
    {
        int * out0 =( output + out_plane_size * (p + 0));
        fill(out0,outh*outw,0);
        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char* r0 = input + in_plane_size*(q+0);
            const signed char* r1 = input + in_plane_size*(q+1);
            const signed char* r2 = input + in_plane_size*(q+2);
            const signed char* r3 = input + in_plane_size*(q+3);
            const signed char* r4 = input + in_plane_size*(q+4);
            const signed char* r5 = input + in_plane_size*(q+5);
            const signed char* r6 = input + in_plane_size*(q+6);
            const signed char* r7 = input + in_plane_size*(q+7);

            const signed char* kernel0 = (const signed char*)mappedWeight + p*inch + q;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                //load inch kernel_0 k0-k7
                asm volatile(
                "vld1.s8    d18, [%0]   \n"
                : "=r"(kernel0) // %0
                : "0" (kernel0)
                :
                );

                asm volatile(
                "0:                            \n"
                        //ld r0-r7
                        "pld        [%2, #64]          \n"
                        "vld1.s8    {d0}, [%2 :64]!    \n"  //r0
                        "pld        [%3, #64]          \n"
                        "vld1.s8    {d1}, [%3 :64]!    \n"  //r1
                        "pld        [%4, #64]          \n"
                        "vld1.s8    {d2}, [%4 :64]!    \n"  //r2
                        "pld        [%5, #64]          \n"
                        "vld1.s8    {d3}, [%5 :64]!    \n"  //r3
                        "pld        [%6, #64]          \n"
                        "vld1.s8    {d4}, [%6 :64]!    \n"  //r4
                        "pld        [%7, #64]          \n"
                        "vld1.s8    {d5}, [%7 :64]!    \n"  //r5
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d6}, [%8 :64]!    \n"  //r6
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d7}, [%9 :64]!    \n"  //r7

                        //load inch kernel_0 k0-k7
                        "vdup.s8    d8, d18[0]          \n"
                        "vdup.s8    d9, d18[1]          \n"
                        "vdup.s8    d10, d18[2]         \n"
                        "vdup.s8    d11, d18[3]         \n"
                        "vdup.s8    d12, d18[4]         \n"
                        "vdup.s8    d13, d18[5]         \n"
                        "vdup.s8    d14, d18[6]         \n"
                        "vdup.s8    d15, d18[7]         \n"

                        //mla
                        "vmull.s8   q14, d0,    d8          \n"
                        "vmlal.s8   q14, d1,    d9          \n"
                        "vmlal.s8   q14, d2,    d10         \n"
                        "vmlal.s8   q14, d3,    d11         \n"
                        "vmlal.s8   q14, d4,    d12         \n"
                        "vmlal.s8   q14, d5,    d13         \n"
                        "vmlal.s8   q14, d6,    d14         \n"
                        "vmlal.s8   q14, d7,    d15         \n"

                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d20-d23}, [%1]     \n" //outptr0_s32
                        "vaddw.s16   q10, q10, d28      \n"
                        "vaddw.s16   q11, q11, d29      \n"
                        "vst1.32    {d20-d23}, [%1]!    \n"

                        //next
                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),          // %0
                "=r"(outptr0),     // %1
                "=r"(r0),          // %2
                "=r"(r1),          // %3
                "=r"(r2),          // %4
                "=r"(r3),          // %5
                "=r"(r4),          // %6
                "=r"(r5),          // %7
                "=r"(r6),          // %8
                "=r"(r7)           // %9
                : "0"(nn),
                "1"(outptr0),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "5"(r3),
                "6"(r4),
                "7"(r5),
                "8"(r6),
                "9"(r7)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11", "q12", "q13", "q14"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0_s8 = input + in_plane_size*(q);
            const signed char* r0 = img0_s8;

            const signed char* kernel0 = (const signed char*)mappedWeight + p*inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            if (nn > 0)
            {
                asm volatile(
                "0:                             \n"
                        //load r0
                        "pld        [%2, #64]           \n"
                        "vld1.s8    {d8}, [%2 :64]!     \n"

                        //mla
                        "vmull.s8   q10, d8, %6         \n"
                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d12-d15}, [%1]     \n"
                        "vaddw.s16   q6, q6, d20        \n"
                        "vaddw.s16   q7, q7, d21        \n"
                        "vst1.32    {d12-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),             // %0
                "=r"(outptr0),        // %1
                "=r"(r0)              // %2
                : "0"(nn),
                "1"(outptr0),
                "2"(r0),
                "w"(_k0)              // %6
                : "cc", "memory", "q4", "q10", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                int sum0 = (int)*r0 * k0;

                *outptr0 += sum0;
                r0++;
                outptr0++;
            }
        }
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("3 part time = %f ",time);

    frame_free(frame);

    return out_frame;
}

cnn_frame *doFeedForward_CONV_1_1_NEON_S8_LEFT4(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d k=%d k=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * mappedWeight = ((cnn_layer *)layer)->conv_layer->W_S8;

    //将输入帧转存为cpu int8帧

    frame = frame_convert_to_cpu_int8(frame);
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int nn_outch = ((cnn_layer *)layer)->output_c / 4;
    int remain_outch_start = nn_outch * 4;
    int in_plane_size = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);
    int out_plane_size = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);
    int inch=((cnn_layer *)layer)->conv_layer->c;
    int outw=((cnn_layer *)layer)->output_w;
    int outh=((cnn_layer *)layer)->output_h;
    int outch=((cnn_layer *)layer)->output_c;

    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        int * out0 = (output + out_plane_size*(p+0));
        int * out1 = (output + out_plane_size*(p+1));
        int * out2 = (output + out_plane_size*(p+2));
        int * out3 = (output + out_plane_size*(p+3));

        //初始化out
        fill(out0,outh*outw,0);
        fill(out1,outh*outw,0);
        fill(out2,outh*outw,0);
        fill(out3,outh*outw,0);


        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* r0 = input + in_plane_size*(q+0);
            const signed char* r1 = input + in_plane_size*(q+1);
            const signed char* r2 = input + in_plane_size*(q+2);
            const signed char* r3 = input + in_plane_size*(q+3);
            const signed char* r4 = input + in_plane_size*(q+4);
            const signed char* r5 = input + in_plane_size*(q+5);
            const signed char* r6 = input + in_plane_size*(q+6);
            const signed char* r7 = input + in_plane_size*(q+7);

            const signed char* kernel0 =(const signed char*) mappedWeight + inch*(p+0) + q;
            const signed char* kernel1 =(const signed char*) mappedWeight + inch*(p+1) + q;
            const signed char* kernel2 =(const signed char*) mappedWeight + inch*(p+2) + q;
            const signed char* kernel3 =(const signed char*) mappedWeight + inch*(p+3) + q;

            int size = outw * outh;

            int nn = size >> 3;

            asm volatile(
            "vld1.s8    d18, [%0]   \n"
                    "vld1.s8    d19, [%1]   \n"
                    "vld1.s8    d24, [%2]   \n"
                    "vld1.s8    d25, [%3]   \n"
            : "=r"(kernel0), // %0
            "=r"(kernel1), // %1
            "=r"(kernel2), // %2
            "=r"(kernel3)  // %3
            : "0"(kernel0),
            "1"(kernel1),
            "2"(kernel2),
            "3"(kernel3)
            :
            );

            if (nn > 0)
            {
                asm volatile(
                "0:                            \n"
                        //ld r0-r7
                        "pld        [%5, #64]          \n"
                        "vld1.s8    {d0}, [%5 :64]!    \n"  //r0

                        "pld        [%6, #64]          \n"
                        "vld1.s8    {d1}, [%6 :64]!    \n"  //r1

                        "pld        [%7, #64]          \n"
                        "vld1.s8    {d2}, [%7 :64]!    \n"  //r2

                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d3}, [%8 :64]!    \n"  //r3

                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d4}, [%9 :64]!    \n"  //r4

                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d5}, [%10 :64]!   \n"  //r5

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6}, [%11 :64]!   \n"  //r6

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d7}, [%12 :64]!   \n"  //r7
                        //###########################################
                        //load inch kernel_0 k0-k7
                        "vdup.s8    d8, d18[0]          \n"
                        "vdup.s8    d9, d18[1]          \n"
                        "vdup.s8    d10, d18[2]         \n"
                        "vdup.s8    d11, d18[3]         \n"
                        "vdup.s8    d12, d18[4]         \n"
                        "vdup.s8    d13, d18[5]         \n"
                        "vdup.s8    d14, d18[6]         \n"
                        "vdup.s8    d15, d18[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d20-d23}, [%1:128] \n" //outptr0_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%1:128]!\n"
                        //###########################################
                        //load inch kernel_1 k0-k7
                        "vdup.s8    d8, d19[0]          \n"
                        "vdup.s8    d9, d19[1]          \n"
                        "vdup.s8    d10, d19[2]         \n"
                        "vdup.s8    d11, d19[3]         \n"
                        "vdup.s8    d12, d19[4]         \n"
                        "vdup.s8    d13, d19[5]         \n"
                        "vdup.s8    d14, d19[6]         \n"
                        "vdup.s8    d15, d19[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr1_s32
                        "pld        [%2, #256]          \n"
                        "vld1.32    {d20-d23}, [%2:128] \n" //outptr1_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%2:128]!\n"
                        //############################################
                        //load inch kernel_2 k0-k7
                        "vdup.s8    d8, d24[0]          \n"
                        "vdup.s8    d9, d24[1]          \n"
                        "vdup.s8    d10, d24[2]         \n"
                        "vdup.s8    d11, d24[3]         \n"
                        "vdup.s8    d12, d24[4]         \n"
                        "vdup.s8    d13, d24[5]         \n"
                        "vdup.s8    d14, d24[6]         \n"
                        "vdup.s8    d15, d24[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr2_s32
                        "pld        [%3, #256]          \n"
                        "vld1.32    {d20-d23}, [%3:128] \n" //outptr2_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%3:128]!\n"
                        //#############################################
                        //load inch kernel_3 k0-k7
                        "vdup.s8    d8, d25[0]          \n"
                        "vdup.s8    d9, d25[1]          \n"
                        "vdup.s8    d10, d25[2]         \n"
                        "vdup.s8    d11, d25[3]         \n"
                        "vdup.s8    d12, d25[4]         \n"
                        "vdup.s8    d13, d25[5]         \n"
                        "vdup.s8    d14, d25[6]         \n"
                        "vdup.s8    d15, d25[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr3_s32
                        "pld        [%4, #256]          \n"
                        "vld1.32    {d20-d23}, [%4:128] \n" //outptr3_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%4:128]!\n"

                        //next
                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),          // %0
                "=r"(outptr0),     // %1
                "=r"(outptr1),     // %2
                "=r"(outptr2),     // %3
                "=r"(outptr3),     // %4
                "=r"(r0),          // %5
                "=r"(r1),          // %6
                "=r"(r2),          // %7
                "=r"(r3),          // %8
                "=r"(r4),          // %9
                "=r"(r5),          // %10
                "=r"(r6),          // %11
                "=r"(r7)           // %12
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "6"(r1),
                "7"(r2),
                "8"(r3),
                "9"(r4),
                "10"(r5),
                "11"(r6),
                "12"(r7)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11"
                );
            }

            asm volatile(
            "0:                            \n"
                    //ld r0-r7
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d0}, [%5 :64]     \n"  //r0

                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d1}, [%6 :64]     \n"  //r1

                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d2}, [%7 :64]     \n"  //r2

                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d3}, [%8 :64]     \n"  //r3

                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d4}, [%9 :64]     \n"  //r4

                    "pld        [%10, #64]         \n"
                    "vld1.s8    {d5}, [%10 :64]    \n"  //r5

                    "pld        [%11, #64]         \n"
                    "vld1.s8    {d6}, [%11 :64]    \n"  //r6

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d7}, [%12 :64]    \n"  //r7

                    "add        %5, #4             \n"
                    "add        %6, #4             \n"
                    "add        %7, #4             \n"
                    "add        %8, #4             \n"
                    "add        %9, #4             \n"
                    "add        %10, #4            \n"
                    "add        %11, #4            \n"
                    "add        %12, #4            \n"
                    //###########################################
                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr0_s32
                    "pld        [%1, #128]          \n"
                    "vld1.32    {d20-d21}, [%1:128] \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%1:128]!\n"
                    //###########################################
                    //load inch kernel_1 k0-k7
                    "vdup.s8    d8, d19[0]          \n"
                    "vdup.s8    d9, d19[1]          \n"
                    "vdup.s8    d10, d19[2]         \n"
                    "vdup.s8    d11, d19[3]         \n"
                    "vdup.s8    d12, d19[4]         \n"
                    "vdup.s8    d13, d19[5]         \n"
                    "vdup.s8    d14, d19[6]         \n"
                    "vdup.s8    d15, d19[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr1_s32
                    "pld        [%2, #128]          \n"
                    "vld1.32    {d20-d21}, [%2:128] \n" //outptr1_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%2:128]!\n"
                    //############################################
                    //load inch kernel_2 k0-k7
                    "vdup.s8    d8, d24[0]          \n"
                    "vdup.s8    d9, d24[1]          \n"
                    "vdup.s8    d10, d24[2]         \n"
                    "vdup.s8    d11, d24[3]         \n"
                    "vdup.s8    d12, d24[4]         \n"
                    "vdup.s8    d13, d24[5]         \n"
                    "vdup.s8    d14, d24[6]         \n"
                    "vdup.s8    d15, d24[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr2_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d20-d21}, [%3:128] \n" //outptr2_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%3:128]!\n"
                    //#############################################
                    //load inch kernel_3 k0-k7
                    "vdup.s8    d8, d25[0]          \n"
                    "vdup.s8    d9, d25[1]          \n"
                    "vdup.s8    d10, d25[2]         \n"
                    "vdup.s8    d11, d25[3]         \n"
                    "vdup.s8    d12, d25[4]         \n"
                    "vdup.s8    d13, d25[5]         \n"
                    "vdup.s8    d14, d25[6]         \n"
                    "vdup.s8    d15, d25[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr3_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d20-d21}, [%4:128] \n" //outptr3_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%4:128]!\n"
            : "=r"(nn),          // %0
            "=r"(outptr0),     // %1
            "=r"(outptr1),     // %2
            "=r"(outptr2),     // %3
            "=r"(outptr3),     // %4
            "=r"(r0),          // %5
            "=r"(r1),          // %6
            "=r"(r2),          // %7
            "=r"(r3),          // %8
            "=r"(r4),          // %9
            "=r"(r5),          // %10
            "=r"(r6),          // %11
            "=r"(r7)           // %12
            : "0"(nn),
            "1"(outptr0),
            "2"(outptr1),
            "3"(outptr2),
            "4"(outptr3),
            "5"(r0),
            "6"(r1),
            "7"(r2),
            "8"(r3),
            "9"(r4),
            "10"(r5),
            "11"(r6),
            "12"(r7)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11"
            );

        }

        //此函数这里没有remain
        for (; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* img0_s8 = input + in_plane_size*q;

            const signed char* kernel0 = mappedWeight + inch*(p+0) + q;
            const signed char* kernel1 = mappedWeight + inch*(p+1) + q;
            const signed char* kernel2 = mappedWeight + inch*(p+2) + q;
            const signed char* kernel3 = mappedWeight + inch*(p+3) + q;

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];

            const signed char* r0 = img0_s8;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);

            if (nn > 0)
            {
                asm volatile(
                "0:                             \n"
                        //load r0
                        "pld        [%5, #64]           \n"
                        "vld1.s8    {d8}, [%5 :64]!     \n"

                        //mla
                        "vmull.s8   q5, d8, %12         \n"
                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d12-d15}, [%1]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%1]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %13         \n"
                        //outptr1_s32
                        "pld        [%2, #256]          \n"
                        "vld1.32    {d12-d15}, [%2]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%2]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %14         \n"
                        //outptr0_s32
                        "pld        [%3, #256]          \n"
                        "vld1.32    {d12-d15}, [%3]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%3]!    \n"

                        //mla
                        "vmull.s8   q5, d8, %15         \n"
                        //outptr0_s32
                        "pld        [%4, #256]          \n"
                        "vld1.32    {d12-d15}, [%4]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),             // %0
                "=r"(outptr0),        // %1
                "=r"(outptr1),        // %2
                "=r"(outptr2),        // %3
                "=r"(outptr3),        // %4
                "=r"(r0)              // %5
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "w"(_k0),             // %12
                "w"(_k1),             // %13
                "w"(_k2),             // %14
                "w"(_k3)              // %15
                : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * k0;
                int sum1 = (int)*r0 * k1;
                int sum2 = (int)*r0 * k2;
                int sum3 = (int)*r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("2 part time = %f ",time);

    t0 = get_timestamp();

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        int * out0 = (output + out_plane_size * (p + 0));

        //bias
        fill(out0,outh*outw,0);
        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char* r0 = input + in_plane_size*(q+0);
            const signed char* r1 = input + in_plane_size*(q+1);
            const signed char* r2 = input + in_plane_size*(q+2);
            const signed char* r3 = input + in_plane_size*(q+3);
            const signed char* r4 = input + in_plane_size*(q+4);
            const signed char* r5 = input + in_plane_size*(q+5);
            const signed char* r6 = input + in_plane_size*(q+6);
            const signed char* r7 = input + in_plane_size*(q+7);

            const signed char* kernel0 = (const signed char*)mappedWeight + p*inch + q;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                //load inch kernel_0 k0-k7
                asm volatile(
                "vld1.s8    d18, [%0]   \n"
                : "=r"(kernel0) // %0
                : "0" (kernel0)
                :
                );

                asm volatile(
                "0:                            \n"
                        //ld r0-r7
                        "pld        [%2, #64]          \n"
                        "vld1.s8    {d0}, [%2 :64]!    \n"  //r0
                        "pld        [%3, #64]          \n"
                        "vld1.s8    {d1}, [%3 :64]!    \n"  //r1
                        "pld        [%4, #64]          \n"
                        "vld1.s8    {d2}, [%4 :64]!    \n"  //r2
                        "pld        [%5, #64]          \n"
                        "vld1.s8    {d3}, [%5 :64]!    \n"  //r3
                        "pld        [%6, #64]          \n"
                        "vld1.s8    {d4}, [%6 :64]!    \n"  //r4
                        "pld        [%7, #64]          \n"
                        "vld1.s8    {d5}, [%7 :64]!    \n"  //r5
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d6}, [%8 :64]!    \n"  //r6
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d7}, [%9 :64]!    \n"  //r7

                        //load inch kernel_0 k0-k7
                        "vdup.s8    d8, d18[0]          \n"
                        "vdup.s8    d9, d18[1]          \n"
                        "vdup.s8    d10, d18[2]         \n"
                        "vdup.s8    d11, d18[3]         \n"
                        "vdup.s8    d12, d18[4]         \n"
                        "vdup.s8    d13, d18[5]         \n"
                        "vdup.s8    d14, d18[6]         \n"
                        "vdup.s8    d15, d18[7]         \n"

                        //mla
                        "vmull.s8   q8, d0, d8          \n"
                        "vmlal.s8   q8, d1, d9          \n"
                        "vmlal.s8   q8, d2, d10         \n"
                        "vmlal.s8   q8, d3, d11         \n"
                        "vmlal.s8   q8, d4, d12         \n"
                        "vmlal.s8   q8, d5, d13         \n"
                        "vmlal.s8   q8, d6, d14         \n"
                        "vmlal.s8   q8, d7, d15         \n"

                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d20-d23}, [%1]     \n" //outptr0_s32
                        "vaddw.s16   q10, q10, d16      \n"
                        "vaddw.s16   q11, q11, d17      \n"
                        "vst1.32    {d20-d23}, [%1]!    \n"

                        //next
                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),          // %0
                "=r"(outptr0),     // %1
                "=r"(r0),          // %2
                "=r"(r1),          // %3
                "=r"(r2),          // %4
                "=r"(r3),          // %5
                "=r"(r4),          // %6
                "=r"(r5),          // %7
                "=r"(r6),          // %8
                "=r"(r7)           // %9
                : "0"(nn),
                "1"(outptr0),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "5"(r3),
                "6"(r4),
                "7"(r5),
                "8"(r6),
                "9"(r7)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q10", "q11", "q12", "q13"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0_s8 = input + in_plane_size*(q);
            const signed char* r0 = img0_s8;

            const signed char* kernel0 =(const signed char*) mappedWeight + p*inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            if (nn > 0)
            {
                asm volatile(
                "0:                             \n"
                        //load r0
                        "pld        [%2, #64]           \n"
                        "vld1.s8    {d8}, [%2 :64]!     \n"

                        //mla
                        "vmull.s8   q5, d8, %6          \n"
                        //outptr0_s32
                        "pld        [%1, #256]          \n"
                        "vld1.32    {d12-d15}, [%1]     \n"
                        "vaddw.s16   q6, q6, d10        \n"
                        "vaddw.s16   q7, q7, d11        \n"
                        "vst1.32    {d12-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                : "=r"(nn),             // %0
                "=r"(outptr0),        // %1
                "=r"(r0)              // %2
                : "0"(nn),
                "1"(outptr0),
                "2"(r0),
                "w"(_k0)              // %6
                : "cc", "memory", "q4", "q5", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                int sum0 = (int)*r0 * k0;

                *outptr0 += sum0;

                r0++;
                outptr0++;
            }
        }
    }
    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("3 part time = %f ",time);

    frame_free(frame);

    return out_frame;

}

cnn_frame *doFeedForward_CONV_1_1_NEON_INT8(cnn_frame *frame, void *layer) {


    int outw=((cnn_layer *)layer)->output_w;
    int outh=((cnn_layer *)layer)->output_h;


    int size = outh * outw; //按输出计算
    int remain = size & 7;

    typedef cnn_frame* (*conv_func_int8)(cnn_frame *frame, void *layer);

    conv_func_int8 conv_func_table[8] =
            {
                    doFeedForward_CONV_1_1_NEON_S8,          //0
                    doFeedForward_CONV_1_1_NEON_S8,          //1
                    doFeedForward_CONV_1_1_NEON_S8,          //2
                    doFeedForward_CONV_1_1_NEON_S8,          //3
                    doFeedForward_CONV_1_1_NEON_S8_LEFT4,    //4   size%8=4
//                    doFeedForward_CONV_1_1_NEON_S8,
                    doFeedForward_CONV_1_1_NEON_S8,          //5
                    doFeedForward_CONV_1_1_NEON_S8,          //6
                    doFeedForward_CONV_1_1_NEON_S8,          //7
            };

    conv_func_int8 conv = conv_func_table[remain];

    cnn_frame *out_frame;
    out_frame= conv(frame, layer);

    return out_frame;
}

cnn_frame *doFeedForward_CONV_DW_S1_NEON_INT8(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int num_threads = ((cnn_layer *)layer)->num_threads;
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d k=%d k=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * _kernel = ((cnn_layer *)layer)->conv_layer->W_S8;


    //将输入帧转存为cpu int8帧

    frame = frame_convert_to_cpu_int8(frame);
    frame = frame_cpu_pad(frame);
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);
    int inputstep = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);

    int outw = ((cnn_layer *)layer)->output_w;
    int outh = ((cnn_layer *)layer)->output_h;
    int outch =  ((cnn_layer *)layer)->output_c;
    int w = frame->w;
    //填充bias

    //输出空间指针
    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int p = 0; p < outch; p++)
    {
        int * out = (output + p*outputstep);

        const signed char* kernel = _kernel + p*9;

        int* outptr0 = out;
        int* outptr0n = outptr0 + outw;

        const signed char* img0 = input + inputstep*p;

        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;
        const signed char* r2 = img0 + w*2;
        const signed char* r3 = img0 + w*3;

        int i = 0;

        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
        for (; i+1 < outh; i+=2) {
            int nn = outw >> 3;
            int remain = outw & 7;

            if (nn > 0)
            {
                asm volatile(
                "0:                              \n"
                        // r0
                        "vld1.s8    {d30-d31}, [%3]      \n"// r0
                        "add    %3, %3, #8               \n"

                        "vext.s8    d10, d30, d31, #1    \n"
                        "vext.s8    d12, d30, d31, #2    \n"

                        "vmovl.s8    q15, d30            \n"// r00
                        "vmovl.s8    q5, d10             \n"// r01
                        "vmovl.s8    q6, d12             \n"// r02
                        // sum0
                        "vmull.s16  q7, d30, %P14[0]     \n"// (r00 - r07) * k00
                        "vmull.s16  q8, d31, %P14[0]     \n"
                        "vmull.s16  q9, d10, %P14[1]     \n"// (r01 - r08) * k01
                        "vmull.s16  q10, d11, %P14[1]    \n"
                        "vmlal.s16  q7, d12, %P14[2]     \n"// (r02 - r09) * k02
                        "vmlal.s16  q8, d13, %P14[2]     \n"

                        // r1
                        "vld1.s8    {d30-d31}, [%4]      \n"// r1
                        "add    %4, %4, #8               \n"

                        "vext.s8    d10, d30, d31, #1    \n"
                        "vext.s8    d12, d30, d31, #2    \n"

                        "vmovl.s8    q15, d30            \n"// r10
                        "vmovl.s8    q5, d10             \n"// r11
                        "vmovl.s8    q6, d12             \n"// r12
                        // sum0
                        "vmlal.s16  q7, d30, %P14[3]     \n"// (r10 - r17) * k03
                        "vmlal.s16  q8, d31, %P14[3]     \n"
                        "vmlal.s16  q9, d10, %P15[0]     \n"// (r11 - r18) * k04
                        "vmlal.s16  q10, d11, %P15[0]    \n"
                        "vmlal.s16  q7, d12, %P15[1]     \n"// (r12 - r19) * k05
                        "vmlal.s16  q8, d13, %P15[1]     \n"
                        // sum1
                        "vmull.s16  q11, d30, %P14[0]    \n"// (r10 - r17) * k00
                        "vmull.s16  q12, d31, %P14[0]    \n"
                        "vmull.s16  q13, d10, %P14[1]    \n"// (r11 - r18) * k01
                        "vmull.s16  q14, d11, %P14[1]    \n"
                        "vmlal.s16  q11, d12, %P14[2]    \n"// (r12 - r19) * k02
                        "vmlal.s16  q12, d13, %P14[2]    \n"

                        // r2
                        "vld1.s8    {d30-d31}, [%5]      \n"// r2
                        "add    %5, %5, #8               \n"

                        "vext.s8    d10, d30, d31, #1    \n"
                        "vext.s8    d12, d30, d31, #2    \n"

                        "vmovl.s8    q15, d30            \n"// r20
                        "vmovl.s8    q5, d10             \n"// r21
                        "vmovl.s8    q6, d12             \n"// r22

                        // sum0
                        "vmlal.s16  q7, d30, %P15[2]     \n"// (r20 - r27) * k06
                        "vmlal.s16  q8, d31, %P15[2]     \n"
                        "vmlal.s16  q9, d10, %P15[3]     \n"// (r21 - r28) * k07
                        "vmlal.s16  q10, d11, %P15[3]    \n"
                        "vmlal.s16  q7, d12, %P16[0]     \n"// (r22 - r29) * k08
                        "vmlal.s16  q8, d13, %P16[0]     \n"
                        // sum1
                        "vmlal.s16  q11, d30, %P14[3]    \n"// (r20 - r27) * k03
                        "vmlal.s16  q12, d31, %P14[3]    \n"
                        "vmlal.s16  q13, d10, %P15[0]    \n"// (r21 - r28) * k04
                        "vmlal.s16  q14, d11, %P15[0]    \n"
                        "vmlal.s16  q11, d12, %P15[1]    \n"// (r22 - r29) * k05
                        "vmlal.s16  q12, d13, %P15[1]    \n"

                        // r3
                        "vld1.s8    {d30-d31}, [%6]      \n"// r3
                        "add    %6, %6, #8               \n"

                        "vext.s8    d10, d30, d31, #1    \n"
                        "vext.s8    d12, d30, d31, #2    \n"

                        "vmovl.s8    q15, d30            \n"// r30
                        "vmovl.s8    q5, d10             \n"// r31
                        "vmovl.s8    q6, d12             \n"// r32

                        // sum1
                        "vmlal.s16  q11, d30, %P15[2]    \n"// (r30 - r37) * k06
                        "vmlal.s16  q12, d31, %P15[2]    \n"
                        "vmlal.s16  q13, d10, %P15[3]    \n"// (r31 - r38) * k07
                        "vmlal.s16  q14, d11, %P15[3]    \n"
                        "vmlal.s16  q11, d12, %P16[0]    \n"// (r32 - r39) * k08
                        "vmlal.s16  q12, d13, %P16[0]    \n"

                        "subs   %0, %0, #1               \n"

                        // add and save
                        "vadd.s32    q7, q7, q9          \n"
                        "vadd.s32    q8, q8, q10         \n"
                        "vadd.s32    q11, q11, q13       \n"
                        "vadd.s32    q12, q12, q14       \n"

                        "vst1.s32    {d14-d17}, [%1]!    \n"
                        "vst1.s32    {d22-d25}, [%2]!    \n"

                        "bne    0b                       \n"

                : "=r"(nn),       // %0
                "=r"(outptr0),  // %1
                "=r"(outptr0n), // %2
                "=r"(r0),       // %3
                "=r"(r1),       // %4
                "=r"(r2),       // %5
                "=r"(r3)        // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr0n),
                "3"(r0),
                "4"(r1),
                "5"(r2),
                "6"(r3),
                "w"(_k0123),    // %14
                "w"(_k4567),    // %15
                "w"(_k8xxx)     // %16
                : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }

            for (; remain>0; remain--)
            {
                // TODO NEON
                int sum0 = 0;
                int sum0n = 0;

                sum0 += (int)r0[0] * kernel[0];
                sum0 += (int)r0[1] * kernel[1];
                sum0 += (int)r0[2] * kernel[2];
                sum0 += (int)r1[0] * kernel[3];
                sum0 += (int)r1[1] * kernel[4];
                sum0 += (int)r1[2] * kernel[5];
                sum0 += (int)r2[0] * kernel[6];
                sum0 += (int)r2[1] * kernel[7];
                sum0 += (int)r2[2] * kernel[8];

                sum0n += (int)r1[0] * kernel[0];
                sum0n += (int)r1[1] * kernel[1];
                sum0n += (int)r1[2] * kernel[2];
                sum0n += (int)r2[0] * kernel[3];
                sum0n += (int)r2[1] * kernel[4];
                sum0n += (int)r2[2] * kernel[5];
                sum0n += (int)r3[0] * kernel[6];
                sum0n += (int)r3[1] * kernel[7];
                sum0n += (int)r3[2] * kernel[8];

                *outptr0 = sum0;
                *outptr0n = sum0n;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr0n++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr0n += outw;
        }

        for (; i < outh; i++)
        {
            int nn = outw >> 3;
            int remain = outw & 7;

            if (nn > 0)
            {
                asm volatile(
                "0:                              \n"
                        // r0
                        "vld1.s8    {d30-d31}, [%2]        \n"// r0
                        "add    %2, %2, #8               \n"

                        "vext.s8    d10, d30, d31, #1      \n"
                        "vext.s8    d12, d30, d31, #2      \n"

                        "vmovl.s8    q15, d30              \n"// r00
                        "vmovl.s8    q5, d10             \n"// r01
                        "vmovl.s8    q6, d12             \n"// r02
                        // sum0
                        "vmull.s16  q7, d30, %P10[0]      \n"// (r00 - r07) * k00
                        "vmull.s16  q8, d31, %P10[0]      \n"
                        "vmull.s16  q9, d10, %P10[1]     \n"// (r01 - r08) * k01
                        "vmull.s16  q10, d11, %P10[1]    \n"
                        "vmlal.s16  q7, d12, %P10[2]     \n"// (r02 - r09) * k02
                        "vmlal.s16  q8, d13, %P10[2]     \n"

                        // r1
                        "vld1.s8    {d30-d31}, [%3]        \n"// r1
                        "add    %3, %3, #8               \n"

                        "vext.s8    d10, d30, d31, #1      \n"
                        "vext.s8    d12, d30, d31, #2      \n"

                        "vmovl.s8    q15, d30              \n"// r10
                        "vmovl.s8    q5, d10             \n"// r11
                        "vmovl.s8    q6, d12             \n"// r12
                        // sum0
                        "vmlal.s16  q7, d30, %P10[3]      \n"// (r10 - r17) * k03
                        "vmlal.s16  q8, d31, %P10[3]      \n"
                        "vmlal.s16  q9, d10, %P11[0]     \n"// (r11 - r18) * k04
                        "vmlal.s16  q10, d11, %P11[0]    \n"
                        "vmlal.s16  q7, d12, %P11[1]     \n"// (r12 - r19) * k05
                        "vmlal.s16  q8, d13, %P11[1]     \n"

                        // r2
                        "vld1.s8    {d30-d31}, [%4]        \n"// r2
                        "add    %4, %4, #8               \n"

                        "vext.s8    d10, d30, d31, #1      \n"
                        "vext.s8    d12, d30, d31, #2      \n"

                        "vmovl.s8    q15, d30              \n"// r20
                        "vmovl.s8    q5, d10             \n"// r21
                        "vmovl.s8    q6, d12             \n"// r22

                        // sum0
                        "vmlal.s16  q7, d30, %P11[2]      \n"// (r20 - r27) * k06
                        "vmlal.s16  q8, d31, %P11[2]      \n"
                        "vmlal.s16  q9, d10, %P11[3]     \n"// (r21 - r28) * k07
                        "vmlal.s16  q10, d11, %P11[3]    \n"
                        "vmlal.s16  q7, d12, %P12[0]     \n"// (r22 - r29) * k08
                        "vmlal.s16  q8, d13, %P12[0]     \n"

                        "subs   %0, %0, #1               \n"

                        // add and save
                        "vadd.s32    q7, q7, q9          \n"
                        "vadd.s32    q8, q8, q10         \n"

                        "vst1.s32    {d14-d17}, [%1]!    \n"

                        "bne    0b                       \n"

                : "=r"(nn),       // %0
                "=r"(outptr0),  // %1
                "=r"(r0),       // %2
                "=r"(r1),       // %3
                "=r"(r2)        // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "w"(_k0123),    // %10
                "w"(_k4567),    // %11
                "w"(_k8xxx)     // %12
                : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }

            for (; remain>0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr0 = sum;

                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("dw s1 conv time = %f ",time);

    frame_free(frame);

    return out_frame;

}

cnn_frame *doFeedForward_CONV_DW_S2_NEON_INT8(cnn_frame *frame, void *layer){

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int num_threads = ((cnn_layer *)layer)->num_threads;
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d k=%d k=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * _kernel = ((cnn_layer *)layer)->conv_layer->W_S8;

    //将输入帧转存为cpu int8帧
    frame = frame_convert_to_cpu_int8(frame);
    frame = frame_cpu_pad(frame);        //这里有疑问
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);

    int inputstep = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int outch = out_frame->c;
    int w = frame->w;
    const int tailstep = w - 2*outw + w;

    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();


#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int p=0; p<outch; p++)
    {
        int * out = (int *)(output + p*outputstep);

        const signed char* kernel = _kernel + p*9;

        int* outptr = out;

        const signed char* img = input + p*inputstep;

        const signed char* r0 = img;
        const signed char* r1 = img + w;
        const signed char* r2 = img + w*2;

        int i = 0;

        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);

        for (; i < outh; i++)
        {
            int nn = outw >> 3;
            int remain = outw & 7;

            if (nn > 0)
            {
                asm volatile(
                "0:                              \n"
                        // r0
                        "vld2.s8    {d30-d31}, [%2]!     \n"// r0
                        "vld2.s8    {d10-d11}, [%2]      \n"
                        "vext.s8    d12, d30, d10, #1    \n"

                        "vmovl.s8    q5, d31             \n"// r01
                        "vmovl.s8    q15, d30            \n"// r00
                        "vmovl.s8    q6, d12             \n"// r02
                        // sum0
                        "vmull.s16  q7, d30, %P10[0]     \n"// (r00 - r07) * k00
                        "vmull.s16  q8, d31, %P10[0]     \n"
                        "vmull.s16  q9, d10, %P10[1]     \n"// (r01 - r08) * k01
                        "vmull.s16  q10, d11, %P10[1]    \n"
                        "vmlal.s16  q7, d12, %P10[2]     \n"// (r02 - r09) * k02
                        "vmlal.s16  q8, d13, %P10[2]     \n"

                        // r1
                        "vld2.s8    {d30-d31}, [%3]!     \n"// r1
                        "vld2.s8    {d10-d11}, [%3]      \n"
                        "vext.s8    d12, d30, d10, #1    \n"

                        "vmovl.s8    q5, d31             \n"// r11
                        "vmovl.s8    q15, d30            \n"// r10
                        "vmovl.s8    q6, d12             \n"// r12
                        // sum0
                        "vmlal.s16  q7, d30, %P10[3]     \n"// (r10 - r17) * k03
                        "vmlal.s16  q8, d31, %P10[3]     \n"
                        "vmlal.s16  q9, d10, %P11[0]     \n"// (r11 - r18) * k04
                        "vmlal.s16  q10, d11, %P11[0]    \n"
                        "vmlal.s16  q7, d12, %P11[1]     \n"// (r12 - r19) * k05
                        "vmlal.s16  q8, d13, %P11[1]     \n"

                        // r2
                        "vld2.s8    {d30-d31}, [%4]!     \n"// r2
                        "vld2.s8    {d10-d11}, [%4]      \n"
                        "vext.s8    d12, d30, d10, #1    \n"

                        "vmovl.s8    q5, d31             \n"// r21
                        "vmovl.s8    q15, d30            \n"// r20
                        "vmovl.s8    q6, d12             \n"// r22

                        // sum0
                        "vmlal.s16  q7, d30, %P11[2]     \n"// (r20 - r27) * k06
                        "vmlal.s16  q8, d31, %P11[2]     \n"
                        "vmlal.s16  q9, d10, %P11[3]     \n"// (r21 - r28) * k07
                        "vmlal.s16  q10, d11, %P11[3]    \n"
                        "vmlal.s16  q7, d12, %P12[0]     \n"// (r22 - r29) * k08
                        "vmlal.s16  q8, d13, %P12[0]     \n"

                        "subs   %0, %0, #1               \n"

                        // add and save
                        "vadd.s32    q7, q7, q9          \n"
                        "vadd.s32    q8, q8, q10         \n"

                        "vst1.s32    {d14-d17}, [%1]!    \n"

                        "bne    0b                       \n"

                : "=r"(nn),       // %0
                "=r"(outptr),   // %1
                "=r"(r0),       // %2
                "=r"(r1),       // %3
                "=r"(r2)        // %4
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "w"(_k0123),    // %10
                "w"(_k4567),    // %11
                "w"(_k8xxx)     // %12
                : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain>0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("3*3 s2 conv time = %f ",time);

    frame_free(frame);

    return out_frame;

}

cnn_frame *doFeedForward_CONV_3_3_S2_NEON_INT8(cnn_frame *frame, void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int num_threads = ((cnn_layer *)layer)->num_threads;
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d w=%d h=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * kernel = ((cnn_layer *)layer)->conv_layer->W_S8;

    //将输入帧转存为cpu int8帧

    frame = frame_convert_to_cpu_int8(frame);
    frame = frame_cpu_pad(frame);
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);
    int inputstep = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int outch = out_frame->c;
    int w = frame->w;
    int h = frame->h;
    int inch = frame->c;
    const int tailstep = w - 2*outw + w;

    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

#pragma omp parallel for
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int *out0 = output + p*outputstep;
        int *out1 = output + (p+1)*outputstep;

        fill(out0,outh*outw,0);
        fill(out1,outh*outw,0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char*)kernel + (p + 1) * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;

            const signed char* img0 = input + q*inputstep;

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;

            const signed char* k10 = kernel1;
            const signed char* k11 = kernel1 + 3;
            const signed char* k12 = kernel1 + 6;

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                "vld1.s8    {d22-d23}, [%0]    \n"   //这里加载16个weight参数 只用九个
                        "vld1.s8    {d24-d25}, [%1]    \n"   //第二个卷积核 一次计算一个通道输入与两个卷积核计算
                : "=r"(kernel0), // %0
                "=r"(kernel1)  // %1
                : "0"(kernel0),
                "1"(kernel1)
                : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                    "0:                             \n" //计算8个输出 取17个即可
                            "pld        [%3, #192]          \n" //这里预取24个 char
                            "vld2.s8    {d0-d1}, [%3]!      \n" // r0  交叉存取 共16个
                            "vld2.s8    {d2-d3}, [%3]       \n" // r0  交叉存取 共16个
                            "vext.8     d3, d0, d2, #1      \n" //第17个输入参数

                            "vdup.s8    d26, d22[0]         \n" //kernel 00  复制8个
                            "vdup.s8    d27, d22[1]         \n" //kernel 01  复制8个
                            "vdup.s8    d28, d22[2]         \n" //kernel 02  复制8个
                            "vmull.s8   q2, d0, d26         \n" // k00  变长乘法
                            "vmlal.s8   q2, d1, d27         \n" // k01  变长乘加
                            "vmlal.s8   q2, d3, d28         \n" // k02  变长乘加

                            "pld        [%4, #192]          \n"
                            "vld2.s8    {d6-d7}, [%4]!      \n" // r1
                            "vld2.s8    {d8-d9}, [%4]       \n"
                            "vext.8     d9, d6, d8, #1      \n"

                            "vdup.s8    d26, d22[3]         \n"
                            "vdup.s8    d27, d22[4]         \n"
                            "vdup.s8    d28, d22[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "pld        [%5, #192]          \n"
                            "vld2.s8    {d10-d11}, [%5]!    \n" // r2
                            "vld2.s8    {d12-d13}, [%5]     \n"
                            "vext.8     d13, d10, d12, #1   \n"

                            "vdup.s8    d26, d22[6]         \n"
                            "vdup.s8    d27, d22[7]         \n"
                            "vdup.s8    d28, d23[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07   q2 16x8
                            "vmlal.s8   q2, d13, d28        \n" // k08   第一层与第一个卷积核

                            "pld        [%1, #256]          \n"
                            "vld1.32    {d14-d17}, [%1]     \n" //sum0 outptr0 8个int32
                            "vaddw.s16   q7, q7, d4         \n" //宽指令 四字和双字 ->四字  d4 16x4  q7 32x4
                            "vaddw.s16   q8, q8, d5         \n"
                            "vst1.32    {d14-d17}, [%1]!    \n" //存储

                            "vdup.s8    d26, d24[0]         \n"
                            "vdup.s8    d27, d24[1]         \n"
                            "vdup.s8    d28, d24[2]         \n"
                            "vmull.s8   q2, d0, d26         \n" // k00
                            "vmlal.s8   q2, d1, d27         \n" // k01
                            "vmlal.s8   q2, d3, d28         \n" // k02

                            "vdup.s8    d26, d24[3]         \n"
                            "vdup.s8    d27, d24[4]         \n"
                            "vdup.s8    d28, d24[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "vdup.s8    d26, d24[6]         \n"
                            "vdup.s8    d27, d24[7]         \n"
                            "vdup.s8    d28, d25[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07
                            "vmlal.s8   q2, d13, d28        \n" // k08

                            "pld        [%2, #256]          \n"
                            "vld1.32    {d14-d17}, [%2]     \n" //sum1
                            "vaddw.s16   q7, q7, d4         \n"
                            "vaddw.s16   q8, q8, d5         \n"
                            "vst1.32    {d14-d17}, [%2]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),        // %1
                    "=r"(outptr1),        // %2
                    "=r"(r0),             // %3
                    "=r"(r1),             // %4
                    "=r"(r2)              // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                    "pld        [%3, #192]          \n"
                            "vld2.s8    {d0-d1}, [%3]!      \n" // r0
                            "vld2.s8    {d2-d3}, [%3]       \n"
                            "vext.8     d3, d0, d2, #1      \n"

                            "vdup.s8    d26, d22[0]         \n"
                            "vdup.s8    d27, d22[1]         \n"
                            "vdup.s8    d28, d22[2]         \n"
                            "vmull.s8   q2, d0, d26         \n" // k00
                            "vmlal.s8   q2, d1, d27         \n" // k01
                            "vmlal.s8   q2, d3, d28         \n" // k02

                            "pld        [%4, #192]          \n"
                            "vld2.s8    {d6-d7}, [%4]!      \n" // r1
                            "vld2.s8    {d8-d9}, [%4]       \n"
                            "vext.8     d9, d6, d8, #1      \n"

                            "vdup.s8    d26, d22[3]         \n"
                            "vdup.s8    d27, d22[4]         \n"
                            "vdup.s8    d28, d22[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "pld        [%5, #192]          \n"
                            "vld2.s8    {d10-d11}, [%5]!    \n" // r2
                            "vld2.s8    {d12-d13}, [%5]     \n"
                            "vext.8     d13, d10, d12, #1   \n"

                            "sub        %3, #8              \n"
                            "sub        %4, #8              \n"
                            "sub        %5, #8              \n"

                            "vdup.s8    d26, d22[6]         \n"
                            "vdup.s8    d27, d22[7]         \n"
                            "vdup.s8    d28, d23[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07
                            "vmlal.s8   q2, d13, d28        \n" // k08

                            "pld        [%1, #128]          \n"
                            "vld1.32    {d14-d15}, [%1]     \n" //sum0
                            "vaddw.s16   q7, q7, d4         \n"
                            "vst1.32    {d14-d15}, [%1]!    \n"

                            "vdup.s8    d26, d24[0]         \n"
                            "vdup.s8    d27, d24[1]         \n"
                            "vdup.s8    d28, d24[2]         \n"
                            "vmull.s8   q2, d0, d26         \n" // k00
                            "vmlal.s8   q2, d1, d27         \n" // k01
                            "vmlal.s8   q2, d3, d28         \n" // k02

                            "vdup.s8    d26, d24[3]         \n"
                            "vdup.s8    d27, d24[4]         \n"
                            "vdup.s8    d28, d24[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "vdup.s8    d26, d24[6]         \n"
                            "vdup.s8    d27, d24[7]         \n"
                            "vdup.s8    d28, d25[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07
                            "vmlal.s8   q2, d13, d28        \n" // k08

                            "pld        [%2, #128]          \n"
                            "vld1.32    {d14-d15}, [%2]     \n" //sum1
                            "vaddw.s16   q7, q7, d4         \n"
                            "vst1.32    {d14-d15}, [%2]!    \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),        // %1
                    "=r"(outptr1),        // %2
                    "=r"(r0),             // %3
                    "=r"(r1),             // %4
                    "=r"(r2)              // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum1 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum1 += (int)r0[0] * kernel1[0];
                    sum1 += (int)r0[1] * kernel1[1];
                    sum1 += (int)r0[2] * kernel1[2];
                    sum1 += (int)r1[0] * kernel1[3];
                    sum1 += (int)r1[1] * kernel1[4];
                    sum1 += (int)r1[2] * kernel1[5];
                    sum1 += (int)r2[0] * kernel1[6];
                    sum1 += (int)r2[1] * kernel1[7];
                    sum1 += (int)r2[2] * kernel1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;   //stride = 2
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
            kernel1 += 9;
        }
    }

#pragma omp parallel for
    for (int p=remain_outch_start; p<outch; p++)
    {
        int *out0 = output + p*outputstep;

        fill(out0,outh*outw,0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = input + q*inputstep;

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                "vld1.s8    {d22-d23}, [%0]    \n"
                : "=r"(kernel0) // %0
                : "0"(kernel0)
                : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                    "0:                             \n"
                            "pld        [%2, #192]          \n"
                            "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                            "vld2.s8    {d2-d3}, [%2]       \n"
                            "vext.8     d3, d0, d2, #1      \n"

                            "vdup.s8    d26, d22[0]         \n"
                            "vdup.s8    d27, d22[1]         \n"
                            "vdup.s8    d28, d22[2]         \n"
                            "vmull.s8   q2, d0, d26         \n" // k00
                            "vmlal.s8   q2, d1, d27         \n" // k01
                            "vmlal.s8   q2, d3, d28         \n" // k02

                            "pld        [%3, #192]          \n"
                            "vld2.s8    {d6-d7}, [%3]!      \n" // r1
                            "vld2.s8    {d8-d9}, [%3]       \n"
                            "vext.8     d9, d6, d8, #1      \n"

                            "vdup.s8    d26, d22[3]         \n"
                            "vdup.s8    d27, d22[4]         \n"
                            "vdup.s8    d28, d22[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "pld        [%4, #192]          \n"
                            "vld2.s8    {d10-d11}, [%4]!    \n" // r2
                            "vld2.s8    {d12-d13}, [%4]     \n"
                            "vext.8     d13, d10, d12, #1   \n"

                            "vdup.s8    d26, d22[6]         \n"
                            "vdup.s8    d27, d22[7]         \n"
                            "vdup.s8    d28, d23[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07
                            "vmlal.s8   q2, d13, d28        \n" // k08

                            "pld        [%1, #256]          \n"
                            "vld1.32    {d14-d17}, [%1]     \n" //sum0
                            "vaddw.s16   q7, q7, d4         \n"
                            "vaddw.s16   q8, q8, d5         \n"
                            "vst1.32    {d14-d17}, [%1]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),    // %1
                    "=r"(r0),             // %2
                    "=r"(r1),             // %3
                    "=r"(r2)              // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13", "q14"
                    );
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                    "pld        [%2, #192]          \n"
                            "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                            "vld2.s8    {d2-d3}, [%2]       \n"
                            "vext.8     d3, d0, d2, #1      \n"

                            "vdup.s8    d26, d22[0]         \n"
                            "vdup.s8    d27, d22[1]         \n"
                            "vdup.s8    d28, d22[2]         \n"
                            "vmull.s8   q2, d0, d26         \n" // k00
                            "vmlal.s8   q2, d1, d27         \n" // k01
                            "vmlal.s8   q2, d3, d28         \n" // k02

                            "pld        [%3, #192]          \n"
                            "vld2.s8    {d6-d7}, [%3]!      \n" // r1
                            "vld2.s8    {d8-d9}, [%3]       \n"
                            "vext.8     d9, d6, d8, #1      \n"

                            "vdup.s8    d26, d22[3]         \n"
                            "vdup.s8    d27, d22[4]         \n"
                            "vdup.s8    d28, d22[5]         \n"
                            "vmlal.s8   q2, d6, d26         \n" // k03
                            "vmlal.s8   q2, d7, d27         \n" // k04
                            "vmlal.s8   q2, d9, d28         \n" // k05

                            "pld        [%4, #192]          \n"
                            "vld2.s8    {d10-d11}, [%4]!    \n" // r2
                            "vld2.s8    {d12-d13}, [%4]     \n"
                            "vext.8     d13, d10, d12, #1   \n"

                            "sub        %2, #8              \n"
                            "sub        %3, #8              \n"
                            "sub        %4, #8              \n"

                            "vdup.s8    d26, d22[6]         \n"
                            "vdup.s8    d27, d22[7]         \n"
                            "vdup.s8    d28, d23[0]         \n"
                            "vmlal.s8   q2, d10, d26        \n" // k06
                            "vmlal.s8   q2, d11, d27        \n" // k07
                            "vmlal.s8   q2, d13, d28        \n" // k08

                            "pld        [%1, #128]          \n"
                            "vld1.32    {d14-d15}, [%1]     \n" //sum0
                            "vaddw.s16   q7, q7, d4         \n"
                            "vst1.32    {d14-d15}, [%1]!    \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),    // %1
                    "=r"(r0),             // %2
                    "=r"(r1),             // %3
                    "=r"(r2)              // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13", "q14"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }

    frame_free(frame);

    return out_frame;
}

/*cnn_frame *doFeedForward_CONV_3_3_S2_NEON_INT8(cnn_frame *frame, void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);
    int num_threads = ((cnn_layer *)layer)->num_threads;
    double time = 0;
    double t0,t1;

    t0 = get_timestamp();

    //获取卷积核
    LOGI("kernel weight size n=%d w=%d h=%d c=%d",((cnn_layer *)layer)->conv_layer->n,((cnn_layer *)layer)->conv_layer->w,((cnn_layer *)layer)->conv_layer->h,((cnn_layer *)layer)->conv_layer->c);
    signed char * kernel = ((cnn_layer *)layer)->conv_layer->W_S8;

    //将输入帧转存为cpu int8帧

    frame = frame_convert_to_cpu_int8(frame);
    frame = frame_cpu_pad(frame);
    signed char *input = frame->data_s8;

    //创建输出帧
    cnn_frame *out_frame = frame_init_output(((cnn_layer *)layer)->output_w, ((cnn_layer *)layer)->output_h, ((cnn_layer *)layer)->output_c, 1);

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(int),16) / sizeof(int);
    int inputstep = alignSize(frame->h*frame->w* sizeof(signed char),16)/ sizeof(signed char);

    int outw = out_frame->w;
    int outh = out_frame->h;
    int outch = out_frame->c;
    int w = frame->w;
    int h = frame->h;
    int inch = frame->c;
    const int tailstep = w - 2*outw + w;

    int *output = (int *)out_frame->data_s8;

    t1 = get_timestamp();
    time = (t1 - t0) / 1000.0L;
    LOGI("data_pre time = %f ",time);

    t0 = get_timestamp();

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

#pragma omp parallel for
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* out0 = output + p*outputstep;
        int* out1 = output + (p+1)*outputstep;

        fill(out0,outw*outh,0);
        fill(out1,outw*outh,0);


        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char*)kernel + (p + 1) * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;

            const signed char* img0 = input + q*inputstep;

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;

            const signed char* k10 = kernel1;
            const signed char* k11 = kernel1 + 3;
            const signed char* k12 = kernel1 + 6;

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                "vld1.s8    {d22-d23}, [%0]    \n"
                "vld1.s8    {d24-d25}, [%1]    \n"
                : "=r"(kernel0), // %0
                "=r"(kernel1)  // %1
                : "0"(kernel0),
                "1"(kernel1)
                : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                    "0:                             \n"
                    "pld        [%3, #192]          \n"
                    "vld2.s8    {d0-d1}, [%3]!      \n" // r0
                    "vld2.s8    {d2-d3}, [%3]       \n"
                    "vext.8     d3, d0, d2, #1      \n"

                    "vdup.s8    d26, d22[0]         \n"
                    "vdup.s8    d27, d22[1]         \n"
                    "vdup.s8    d28, d22[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "pld        [%4, #192]          \n"
                    "vld2.s8    {d6-d7}, [%4]!      \n" // r1
                    "vld2.s8    {d8-d9}, [%4]       \n"
                    "vext.8     d9, d6, d8, #1      \n"

                    "vdup.s8    d26, d22[3]         \n"
                    "vdup.s8    d27, d22[4]         \n"
                    "vdup.s8    d28, d22[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "pld        [%5, #192]          \n"
                    "vld2.s8    {d10-d11}, [%5]!    \n" // r2
                    "vld2.s8    {d12-d13}, [%5]     \n"
                    "vext.8     d13, d10, d12, #1   \n"

                    "vdup.s8    d26, d22[6]         \n"
                    "vdup.s8    d27, d22[7]         \n"
                    "vdup.s8    d28, d23[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%1, #256]          \n"
                    "vld1.32    {d14-d17}, [%1]     \n" //sum0
                    "vaddw.s16   q7, q7, d4         \n"
                    "vaddw.s16   q8, q8, d5         \n"
                    "vst1.32    {d14-d17}, [%1]!    \n"

                    "vdup.s8    d26, d24[0]         \n"
                    "vdup.s8    d27, d24[1]         \n"
                    "vdup.s8    d28, d24[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "vdup.s8    d26, d24[3]         \n"
                    "vdup.s8    d27, d24[4]         \n"
                    "vdup.s8    d28, d24[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "vdup.s8    d26, d24[6]         \n"
                    "vdup.s8    d27, d24[7]         \n"
                    "vdup.s8    d28, d25[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%2, #256]          \n"
                    "vld1.32    {d14-d17}, [%2]     \n" //sum1
                    "vaddw.s16   q7, q7, d4         \n"
                    "vaddw.s16   q8, q8, d5         \n"
                    "vst1.32    {d14-d17}, [%2]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),        // %1
                    "=r"(outptr1),        // %2
                    "=r"(r0),             // %3
                    "=r"(r1),             // %4
                    "=r"(r2)              // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                    "pld        [%3, #192]          \n"
                    "vld2.s8    {d0-d1}, [%3]!      \n" // r0
                    "vld2.s8    {d2-d3}, [%3]       \n"
                    "vext.8     d3, d0, d2, #1      \n"

                    "vdup.s8    d26, d22[0]         \n"
                    "vdup.s8    d27, d22[1]         \n"
                    "vdup.s8    d28, d22[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "pld        [%4, #192]          \n"
                    "vld2.s8    {d6-d7}, [%4]!      \n" // r1
                    "vld2.s8    {d8-d9}, [%4]       \n"
                    "vext.8     d9, d6, d8, #1      \n"

                    "vdup.s8    d26, d22[3]         \n"
                    "vdup.s8    d27, d22[4]         \n"
                    "vdup.s8    d28, d22[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "pld        [%5, #192]          \n"
                    "vld2.s8    {d10-d11}, [%5]!    \n" // r2
                    "vld2.s8    {d12-d13}, [%5]     \n"
                    "vext.8     d13, d10, d12, #1   \n"

                    "sub        %3, #8              \n"
                    "sub        %4, #8              \n"
                    "sub        %5, #8              \n"

                    "vdup.s8    d26, d22[6]         \n"
                    "vdup.s8    d27, d22[7]         \n"
                    "vdup.s8    d28, d23[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%1, #128]          \n"
                    "vld1.32    {d14-d15}, [%1]     \n" //sum0
                    "vaddw.s16   q7, q7, d4         \n"
                    "vst1.32    {d14-d15}, [%1]!    \n"

                    "vdup.s8    d26, d24[0]         \n"
                    "vdup.s8    d27, d24[1]         \n"
                    "vdup.s8    d28, d24[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "vdup.s8    d26, d24[3]         \n"
                    "vdup.s8    d27, d24[4]         \n"
                    "vdup.s8    d28, d24[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "vdup.s8    d26, d24[6]         \n"
                    "vdup.s8    d27, d24[7]         \n"
                    "vdup.s8    d28, d25[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%2, #128]          \n"
                    "vld1.32    {d14-d15}, [%2]     \n" //sum1
                    "vaddw.s16   q7, q7, d4         \n"
                    "vst1.32    {d14-d15}, [%2]!    \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),        // %1
                    "=r"(outptr1),        // %2
                    "=r"(r0),             // %3
                    "=r"(r1),             // %4
                    "=r"(r2)              // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum1 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum1 += (int)r0[0] * kernel1[0];
                    sum1 += (int)r0[1] * kernel1[1];
                    sum1 += (int)r0[2] * kernel1[2];
                    sum1 += (int)r1[0] * kernel1[3];
                    sum1 += (int)r1[1] * kernel1[4];
                    sum1 += (int)r1[2] * kernel1[5];
                    sum1 += (int)r2[0] * kernel1[6];
                    sum1 += (int)r2[1] * kernel1[7];
                    sum1 += (int)r2[2] * kernel1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
            kernel1 += 9;
        }
    }

#pragma omp parallel for
    for (int p = remain_outch_start; p<outch; p++)
    {
        int* out0 = output + p*outputstep;

        fill(out0,outw*outh,0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = input + inputstep*q;

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                "vld1.s8    {d22-d23}, [%0]    \n"
                : "=r"(kernel0) // %0
                : "0"(kernel0)
                : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                    "0:                             \n"
                    "pld        [%2, #192]          \n"
                    "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                    "vld2.s8    {d2-d3}, [%2]       \n"
                    "vext.8     d3, d0, d2, #1      \n"

                    "vdup.s8    d26, d22[0]         \n"
                    "vdup.s8    d27, d22[1]         \n"
                    "vdup.s8    d28, d22[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "pld        [%3, #192]          \n"
                    "vld2.s8    {d6-d7}, [%3]!      \n" // r1
                    "vld2.s8    {d8-d9}, [%3]       \n"
                    "vext.8     d9, d6, d8, #1      \n"

                    "vdup.s8    d26, d22[3]         \n"
                    "vdup.s8    d27, d22[4]         \n"
                    "vdup.s8    d28, d22[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "pld        [%4, #192]          \n"
                    "vld2.s8    {d10-d11}, [%4]!    \n" // r2
                    "vld2.s8    {d12-d13}, [%4]     \n"
                    "vext.8     d13, d10, d12, #1   \n"

                    "vdup.s8    d26, d22[6]         \n"
                    "vdup.s8    d27, d22[7]         \n"
                    "vdup.s8    d28, d23[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%1, #256]          \n"
                    "vld1.32    {d14-d17}, [%1]     \n" //sum0
                    "vaddw.s16   q7, q7, d4         \n"
                    "vaddw.s16   q8, q8, d5         \n"
                    "vst1.32    {d14-d17}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),    // %1
                    "=r"(r0),             // %2
                    "=r"(r1),             // %3
                    "=r"(r2)              // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13", "q14"
                    );
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                    "pld        [%2, #192]          \n"
                    "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                    "vld2.s8    {d2-d3}, [%2]       \n"
                    "vext.8     d3, d0, d2, #1      \n"

                    "vdup.s8    d26, d22[0]         \n"
                    "vdup.s8    d27, d22[1]         \n"
                    "vdup.s8    d28, d22[2]         \n"
                    "vmull.s8   q2, d0, d26         \n" // k00
                    "vmlal.s8   q2, d1, d27         \n" // k01
                    "vmlal.s8   q2, d3, d28         \n" // k02

                    "pld        [%3, #192]          \n"
                    "vld2.s8    {d6-d7}, [%3]!      \n" // r1
                    "vld2.s8    {d8-d9}, [%3]       \n"
                    "vext.8     d9, d6, d8, #1      \n"

                    "vdup.s8    d26, d22[3]         \n"
                    "vdup.s8    d27, d22[4]         \n"
                    "vdup.s8    d28, d22[5]         \n"
                    "vmlal.s8   q2, d6, d26         \n" // k03
                    "vmlal.s8   q2, d7, d27         \n" // k04
                    "vmlal.s8   q2, d9, d28         \n" // k05

                    "pld        [%4, #192]          \n"
                    "vld2.s8    {d10-d11}, [%4]!    \n" // r2
                    "vld2.s8    {d12-d13}, [%4]     \n"
                    "vext.8     d13, d10, d12, #1   \n"

                    "sub        %2, #8              \n"
                    "sub        %3, #8              \n"
                    "sub        %4, #8              \n"

                    "vdup.s8    d26, d22[6]         \n"
                    "vdup.s8    d27, d22[7]         \n"
                    "vdup.s8    d28, d23[0]         \n"
                    "vmlal.s8   q2, d10, d26        \n" // k06
                    "vmlal.s8   q2, d11, d27        \n" // k07
                    "vmlal.s8   q2, d13, d28        \n" // k08

                    "pld        [%1, #128]          \n"
                    "vld1.32    {d14-d15}, [%1]     \n" //sum0
                    "vaddw.s16   q7, q7, d4         \n"
                    "vst1.32    {d14-d15}, [%1]!    \n"
                    : "=r"(nn),             // %0
                    "=r"(outptr0),    // %1
                    "=r"(r0),             // %2
                    "=r"(r1),             // %3
                    "=r"(r2)              // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13", "q14"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}*/

//mix conv
//父子进程混合层   由父进程执行GPU 子进程执行CPU（classifier中单独完成）
//此处为父进程中得GPU执行函数
void doFeedForward_CONV_ASM_NEON_MIX(cnn_frame *frame, void *layer,float *shm2_buffer){

    int outputstep = alignSize(((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w* sizeof(float),16) / sizeof(float);
//    int weightstep = alignSize(((cnn_layer *)layer)->conv_layer->c* sizeof(float),16) / sizeof(float);
    int weightstep = ((cnn_layer *)layer)->conv_layer->c;
    int inputstep = alignSize(frame->h*frame->w* sizeof(float),16)/ sizeof(float);

    float * mappedWeight = ((cnn_layer *)layer)->conv_layer->W + ((cnn_layer *)layer)->conv_layer->n/2*weightstep;

    float * output_data = shm2_buffer;
    ((cnn_layer *)layer)->output_c /= 2;
    //每个卷积核数量维度上循环 C%6
    int n_output = ((cnn_layer *)layer)->output_c / 6 ;//计算一半的量
    int remain_n_output_start = n_output * 6;

    #pragma omp parallel for
    for (int m = 0 ; m < n_output; ++m) {
        //取输出的6个通道平面的数据，并用bias初始化
        int n = m*6;
        float * out0 = output_data + outputstep*(n+0);
        float * out1 = output_data + outputstep*(n+1);
        float * out2 = output_data + outputstep*(n+2);
        float * out3 = output_data + outputstep*(n+3);
        float * out4 = output_data + outputstep*(n+4);
        float * out5 = output_data + outputstep*(n+5);

        //bias初始化out


        int c = 0;
        for (; c+3 < ((cnn_layer *)layer)->conv_layer->c; c+=4) {

            float * outptr0 = out0;
            float * outptr1 = out1;
            float * outptr2 = out2;
            float * outptr3 = out3;
            float * outptr4 = out4;
            float * outptr5 = out5;

            //取输入的4个通道上的数据
            const float* img0 = frame->data + inputstep*(c+0);
            const float* img1 = frame->data + inputstep*(c+1);
            const float* img2 = frame->data + inputstep*(c+2);
            const float* img3 = frame->data + inputstep*(c+3);

            //取6个卷积核，每个卷积核中取通道数上8个数据
            const float* kernel0 = mappedWeight + weightstep*(n+0) + c;
            const float* kernel1 = mappedWeight + weightstep*(n+1) + c;
            const float* kernel2 = mappedWeight + weightstep*(n+2) + c;
            const float* kernel3 = mappedWeight + weightstep*(n+3) + c;
            const float* kernel4 = mappedWeight + weightstep*(n+4) + c;
            const float* kernel5 = mappedWeight + weightstep*(n+5) + c;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = ((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w;
            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);
            float32x4_t _k4 = vld1q_f32(kernel4);
            float32x4_t _k5 = vld1q_f32(kernel5);


            //每次取 4*h*w（C*H*W）的输入，6*4*1*1 的卷积核，得出 6*h*w的输出
            if (nn > 0)
            {
                asm volatile(
                "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                        "vmla.f32   q6, q12, %e22[0]        \n"

                        "0:                                 \n"

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

                        "vmla.f32   q7, q12, %e23[0]        \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

                        "vmla.f32   q8, q12, %e24[0]        \n"

                        "pld        [%8, #128]              \n"
                        "vld1.f32   {d26-d27}, [%8 :128]!   \n"// q13 = r1

                        "vmla.f32   q9, q12, %e25[0]        \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

                        "vmla.f32   q6, q13, %e22[1]        \n"
                        "vmla.f32   q7, q13, %e23[1]        \n"

                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

                        "vmla.f32   q10, q12, %e26[0]       \n"
                        "vmla.f32   q11, q12, %e27[0]       \n"

                        "vmla.f32   q8, q13, %e24[1]        \n"
                        "vmla.f32   q9, q13, %e25[1]        \n"

                        "pld        [%9, #128]              \n"
                        "vld1.f32   {d28-d29}, [%9 :128]!   \n"// q14 = r2

                        "vmla.f32   q10, q13, %e26[1]       \n"
                        "vmla.f32   q11, q13, %e27[1]       \n"

                        "vmla.f32   q6, q14, %f22[0]        \n"
                        "vmla.f32   q7, q14, %f23[0]        \n"
                        "vmla.f32   q8, q14, %f24[0]        \n"
                        "vmla.f32   q9, q14, %f25[0]        \n"

                        "pld        [%10, #128]             \n"
                        "vld1.f32   {d30-d31}, [%10 :128]!  \n"// q15 = r3

                        "vmla.f32   q10, q14, %f26[0]       \n"
                        "vmla.f32   q11, q14, %f27[0]       \n"

                        "vmla.f32   q6, q15, %f22[1]        \n"
                        "vmla.f32   q7, q15, %f23[1]        \n"
                        "vmla.f32   q8, q15, %f24[1]        \n"
                        "vmla.f32   q9, q15, %f25[1]        \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "vmla.f32   q10, q15, %f26[1]       \n"
                        "vmla.f32   q11, q15, %f27[1]       \n"

                        "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                        "vmla.f32   q6, q12, %e22[0]        \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                        "subs       %0, #1                  \n"

                        "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                        "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                        "bne        0b                      \n"

                        "sub        %7, #16                 \n"

                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(outptr4),// %5
                "=r"(outptr5),// %6
                "=r"(r0),     // %7
                "=r"(r1),     // %8
                "=r"(r2),     // %9
                "=r"(r3)      // %10
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(r0),
                "8"(r1),
                "9"(r2),
                "10"(r3),
                "w"(_k0),     // %22
                "w"(_k1),     // %23
                "w"(_k2),     // %24
                "w"(_k3),     // %25
                "w"(_k4),     // %26
                "w"(_k5)      // %27
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }

            //(w*h)%4 的部分计算
            for(;remain>0; remain--)
            {
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3];
                float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }

        //C%4 的部分计算
        for (; c < ((cnn_layer *)layer)->conv_layer->c; c++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;

            const float* img0 = frame->data + inputstep*c;

            const float* kernel0 = mappedWeight +weightstep*(n+0) + c;
            const float* kernel1 = mappedWeight + weightstep*(n+1) + c;
            const float* kernel2 = mappedWeight + weightstep*(n+2) + c;
            const float* kernel3 = mappedWeight + weightstep*(n+3) + c;
            const float* kernel4 = mappedWeight + weightstep*(n+4) + c;
            const float* kernel5 = mappedWeight + weightstep*(n+5) + c;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];
            const float k4 = kernel4[0];
            const float k5 = kernel5[0];

            const float* r0 = img0;

            int size = ((cnn_layer *)layer)->output_h*((cnn_layer *)layer)->output_w;

            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
            float32x4_t _k4 = vdupq_n_f32(k4);
            float32x4_t _k5 = vdupq_n_f32(k5);

            if (nn > 0)
            {
                asm volatile(
                "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "0:                                 \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

                        "vmla.f32   q6, q12, %q16           \n"// outptr0 += r0*_k0

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

                        "vmla.f32   q7, q12, %q17           \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

                        "vmla.f32   q8, q12, %q18           \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

                        "vmla.f32   q9, q12, %q19           \n"

                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

                        "vmla.f32   q10, q12, %q20          \n"
                        "vmla.f32   q11, q12, %q21          \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

                        "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

                        "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                        "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                        "bne        0b                      \n"

                        "sub        %7, #16                 \n"

                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(outptr4),// %5
                "=r"(outptr5),// %6
                "=r"(r0)      // %7
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(r0),
                "w"(_k0),     // %16
                "w"(_k1),     // %17
                "w"(_k2),     // %18
                "w"(_k3),     // %19
                "w"(_k4),     // %20
                "w"(_k5)      // %21
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
                );
            }
            for (; remain>0; remain--)
            {
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;
                float sum4 = *r0 * k4;
                float sum5 = *r0 * k5;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }
    }

    n_output = (((cnn_layer *)layer)->output_c - remain_n_output_start) >> 2;

    #pragma omp parallel for
    for (int pp=0; pp<n_output; pp++) {
        int p = remain_n_output_start + pp * 4;

        float *out0 = output_data + outputstep * (p + 0);
        float *out1 = output_data + outputstep * (p + 1);
        float *out2 = output_data + outputstep * (p + 2);
        float *out3 = output_data + outputstep * (p + 3);

        //bias初始化out

        int q = 0;
        for (; q + 3 < ((cnn_layer *)layer)->conv_layer->c; q += 4) {
            float *outptr0 = out0;
            float *outptr1 = out1;
            float *outptr2 = out2;
            float *outptr3 = out3;

            const float *img0 = frame->data + inputstep * (q + 0);
            const float *img1 = frame->data + inputstep * (q + 1);
            const float *img2 = frame->data + inputstep * (q + 2);
            const float *img3 = frame->data + inputstep * (q + 3);

            const float *kernel0 = mappedWeight + (p + 0) * weightstep + q;
            const float *kernel1 = mappedWeight + (p + 1) * weightstep + q;
            const float *kernel2 = mappedWeight + (p + 2) * weightstep + q;
            const float *kernel3 = mappedWeight + (p + 3) * weightstep + q;

            const float *r0 = img0;
            const float *r1 = img1;
            const float *r2 = img2;
            const float *r3 = img3;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);

            if (nn > 0) {
                asm volatile(
                "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "0:                                 \n"

                        "vmla.f32   q8, q6, %e18[0]         \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q9, q7, %e18[0]         \n"

                        "vmla.f32   q10, q6, %e19[0]        \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q11, q7, %e19[0]        \n"

                        "vmla.f32   q12, q6, %e20[0]        \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q13, q7, %e20[0]        \n"

                        "pld        [%6, #256]              \n"
                        "vld1.f32   {d8-d11}, [%6 :128]!    \n"

                        "vmla.f32   q14, q6, %e21[0]        \n"
                        "vmla.f32   q15, q7, %e21[0]        \n"

                        "vmla.f32   q8, q4, %e18[1]         \n"
                        "vmla.f32   q9, q5, %e18[1]         \n"

                        "vmla.f32   q10, q4, %e19[1]        \n"
                        "vmla.f32   q11, q5, %e19[1]        \n"

                        "vmla.f32   q12, q4, %e20[1]        \n"
                        "vmla.f32   q13, q5, %e20[1]        \n"

                        "pld        [%7, #256]              \n"
                        "vld1.f32   {d12-d15}, [%7 :128]!   \n"

                        "vmla.f32   q14, q4, %e21[1]        \n"
                        "vmla.f32   q15, q5, %e21[1]        \n"

                        "vmla.f32   q8, q6, %f18[0]         \n"
                        "vmla.f32   q9, q7, %f18[0]         \n"

                        "vmla.f32   q10, q6, %f19[0]        \n"
                        "vmla.f32   q11, q7, %f19[0]        \n"

                        "vmla.f32   q12, q6, %f20[0]        \n"
                        "vmla.f32   q13, q7, %f20[0]        \n"

                        "pld        [%8, #256]              \n"
                        "vld1.f32   {d8-d11}, [%8 :128]!    \n"

                        "vmla.f32   q14, q6, %f21[0]        \n"
                        "vmla.f32   q15, q7, %f21[0]        \n"

                        "vmla.f32   q8, q4, %f18[1]         \n"
                        "vmla.f32   q9, q5, %f18[1]         \n"

                        "vmla.f32   q10, q4, %f19[1]        \n"
                        "vmla.f32   q11, q5, %f19[1]        \n"

                        "vmla.f32   q12, q4, %f20[1]        \n"
                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "vmla.f32   q13, q5, %f20[1]        \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "vmla.f32   q14, q4, %f21[1]        \n"
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"

                        "vmla.f32   q15, q5, %f21[1]        \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"

                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"

                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(r0),     // %5
                "=r"(r1),     // %6
                "=r"(r2),     // %7
                "=r"(r3)      // %8
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "6"(r1),
                "7"(r2),
                "8"(r3),
                "w"(_k0),     // %18
                "w"(_k1),     // %19
                "w"(_k2),     // %20
                "w"(_k3)      // %21
                : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain > 0; remain--) {
                float sum0 =
                        *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 =
                        *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 =
                        *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 =
                        *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
        for (; q<((cnn_layer *)layer)->conv_layer->c; q++) {
            float *outptr0 = out0;
            float *outptr1 = out1;
            float *outptr2 = out2;
            float *outptr3 = out3;

            const float *img0 = frame->data + inputstep* (q + 0);

            const float *kernel0 = mappedWeight + p * weightstep + q;
            const float *kernel1 = mappedWeight + (p + 1) * weightstep + q;
            const float *kernel2 = mappedWeight + (p + 2) * weightstep + q;
            const float *kernel3 = mappedWeight + (p + 3) * weightstep + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float *r0 = img0;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);

            if (nn > 0) {
                asm volatile(
                "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "0:                                 \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "vmla.f32   q8, q6, %q12            \n"
                        "vmla.f32   q9, q7, %q12            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q10, q6, %q13           \n"
                        "vmla.f32   q11, q7, %q13           \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q12, q6, %q14           \n"
                        "vmla.f32   q13, q7, %q14           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q14, q6, %q15           \n"
                        "vmla.f32   q15, q7, %q15           \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"
                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                : "=r"(nn),     // %0
                "=r"(outptr0),// %1
                "=r"(outptr1),// %2
                "=r"(outptr2),// %3
                "=r"(outptr3),// %4
                "=r"(r0)      // %5
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(r0),
                "w"(_k0),     // %12
                "w"(_k1),     // %13
                "w"(_k2),     // %14
                "w"(_k3)      // %15
                : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            for (; remain > 0; remain--) {
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    remain_n_output_start += n_output << 2;

    #pragma omp parallel for
    for (int p=remain_n_output_start; p<((cnn_layer *)layer)->output_c; p++)
    {
        float * out = output_data + outputstep * p;
        //bias初始化

        int q = 0;

        for (; q+3<((cnn_layer *)layer)->conv_layer->c; q+=4) {
            float *outptr = out;

            const float *img0 = frame->data + inputstep * (q + 0);
            const float *img1 = frame->data + inputstep * (q + 1);
            const float *img2 = frame->data + inputstep * (q + 2);
            const float *img3 = frame->data + inputstep * (q + 3);

            const float *kernel0 = mappedWeight + p * weightstep + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float *r0 = img0;
            const float *r1 = img1;
            const float *r2 = img2;
            const float *r3 = img3;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);

            if (nn > 0) {
                asm volatile(
                "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q12        \n"
                        "vmla.f32   q1, q3, %q12        \n"
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d4-d7}, [%3 :128]! \n"
                        "vmla.f32   q0, q2, %q13        \n"
                        "vmla.f32   q1, q3, %q13        \n"
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d4-d7}, [%4 :128]! \n"
                        "vmla.f32   q0, q2, %q14        \n"
                        "vmla.f32   q1, q3, %q14        \n"
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"
                        "vmla.f32   q0, q2, %q15        \n"
                        "vmla.f32   q1, q3, %q15        \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                "=r"(outptr), // %1
                "=r"(r0),     // %2
                "=r"(r1),     // %3
                "=r"(r2),     // %4
                "=r"(r3)      // %5
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "3"(r1),
                "4"(r2),
                "5"(r3),
                "w"(_k0),     // %12
                "w"(_k1),     // %13
                "w"(_k2),     // %14
                "w"(_k3)      // %15
                : "cc", "memory", "q0", "q1", "q2", "q3"
                );
            }
            for (; remain > 0; remain--) {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }
        }
        for (; q<((cnn_layer *)layer)->conv_layer->c; q++) {
            float *outptr = out;

            const float *img0 = frame->data + inputstep * (q + 0);

            const float *kernel0 = mappedWeight + p * weightstep + q;
            const float k0 = kernel0[0];

            const float *r0 = img0;

            int size = ((cnn_layer *)layer)->output_h * ((cnn_layer *)layer)->output_w;

            int nn = size >> 3;
            int remain = size & 7;

            float32x4_t _k0 = vdupq_n_f32(k0);

            if (nn > 0) {
                asm volatile(
                "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q6         \n"
                        "vmla.f32   q1, q3, %q6         \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                "=r"(outptr), // %1
                "=r"(r0)      // %2
                : "0"(nn),
                "1"(outptr),
                "2"(r0),
                "w"(_k0)      // %6
                : "cc", "memory", "q0", "q1", "q2", "q3"
                );
            }
            for (; remain > 0; remain--) {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
        }
    }

    //激活函数

}

cnn_frame *doFeedForward_CONV_DW_MIX(cnn_frame *frame, void *layer,float *shm1_buffer,int client_fd,int LaterNum) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);
    OpenCLObjects *openCLObjects = getOpenClObject();

    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;
    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;
    double t0,t1,FatherThreadTime;
    cl_kernel kernel;

    t0=get_timestamp();

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);
    // cnn_frame *output0= ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(frame->w+2*conv_layer->pad[0], frame->h+2*conv_layer->pad[0], frame->c) : frame_init_gpu(frame->w+2*conv_layer->pad[0], frame->h+2*conv_layer->pad[0], frame->c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug
    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);
    //debug

    int i = 0;
    kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_dw_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c/2;
//    int gs3 = (output_c > 256 ? 256 : output_c);

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    cl_event event;
    cl_int status;
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error here CONV %d\n",status);
    }


    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);
    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("GPUTime in %f ms\n", FatherThreadTime);
    frame_free(frame);


    //在这里完成cl_half 转cl_float
    t0=get_timestamp();
    cl_mem cl_float_buffer=clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            output->w * output->h * output->c * sizeof(cl_float), //size in bytes
            NULL,//buffer of data
            &err);


    kernel = openCLObjects->convert_half_to_float_kernel.kernel;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output->cl_data);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_float_buffer);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    if(err!=CL_SUCCESS)
    {
        LOGI("error event code : %d \n",err);
        LOGI("error  here clCreateBuffer!!!\n");
    }

    size_t convertSize[1] = {(size_t) output->w * output->h * output->c};
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            convertSize,
            0,
            0, NULL, &event
    );

    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);

    if(err!=CL_SUCCESS)
    {
        LOGI("error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error here MEMCOPY!!! %d\n",status);
    }
    err = clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    float *inputdata= (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					cl_float_buffer, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					 output->w * output->h * output->c * sizeof(float), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("DW Half to Float Time in %f ms\n", FatherThreadTime);
    //cl_half转cl_float代码结束


    //在这里将cl_float转存到共享内存中
    //注意：此处数据存放维度可能不正确！
    int copsize= output->w * output->h* sizeof(float);
    int shm_cstep= alignSize(copsize,16)/sizeof(float);
    int cl_cstep=output->w * output->h;
    t0=get_timestamp();
    //按层将数据复制进共享内存
#pragma omp parallel for
    for (int i = 0; i < output->c; ++i) {
        memcpy(shm1_buffer+shm_cstep*i,inputdata+cl_cstep*i,copsize);
    }

    //释放cl_float空间
    clEnqueueUnmapMemObject(openCLObjects->queue, \
					cl_float_buffer, \
					inputdata, \
					0, NULL, NULL);
    clReleaseMemObject(cl_float_buffer);
    //共享内存复制完成，socket发出同步信号
    char charbuffer[2];
    charbuffer[0]='A'+LaterNum;
    SocketSendALL(charbuffer,1);
    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("DW Copy SHM Time in %f ms\n", FatherThreadTime);

    return output;
}

cnn_frame *doFeedForward_CONV_DW_1_1_MIX(cnn_frame *frame, void *layer,float *shm2_buffer,int client_fd,int LaterNum) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);
    OpenCLObjects *openCLObjects = getOpenClObject();
    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;
    cl_int err = CL_SUCCESS;

    int output_w = ((cnn_layer *)layer)->output_w;
    int output_h = ((cnn_layer *)layer)->output_h;
    int output_c = ((cnn_layer *)layer)->output_c / 2;
    double t0,t1,FatherThreadTime;
    cl_kernel kernel;

    t0=get_timestamp();
    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c*2) : frame_init_gpu(output_w, output_h, output_c*2);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug

    int  i = 0;
    kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_1_1_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
//    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);   //这里为原来的一半
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    int global_x = output->w;
    int global_y = output->h;
//    int gs3 = output_c/2;
    int gs3 = (output_c > 256 ? 256 : output_c);

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    cl_event  event;
    cl_int status;

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);
    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("GPUTime in %f ms\n", FatherThreadTime);
    frame_free(frame);

    //等待子进程算完返回
    char charbuffer[2];

    t0=get_timestamp();
    SocketRecvALL(charbuffer,1);
    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("DW_1_1_GPU WAIT Time in %f ms\n", FatherThreadTime);

    //////////////////////////////////////////////
    //将共享内存中的数据转化到half
    int copsize= output->w * output->h * sizeof(float);
    int shm_cstep= alignSize(copsize,16) / sizeof(float);
    int cl_cstep=output->w * output->h;
    cl_mem cl_data = NULL;
    t0=get_timestamp();
    float *buffer=(float *)malloc(output->w*output->h*output->c/2 * sizeof(float));

    //按层将数据复制进cl_float
#pragma omp parallel for
    for (int i = 0; i < output->c/2; ++i) {
        memcpy(buffer+i*cl_cstep,shm2_buffer+shm_cstep*i,copsize);
    }

    cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            output->w * output->h * output->c/2 * sizeof(float), //size in bytes
            buffer,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    if(err!=CL_SUCCESS)
    {
        LOGI("error  here CreateBuffer!!!\n");
    }

    int output_bias = output->w * output->h * output->c/2;
    kernel = openCLObjects->convert_float_to_half_kernel_bias.kernel;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->cl_data);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_bias);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t convertSize[1] = {(size_t) output->w * output->h * output->c/2};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            convertSize,
            0,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("frame_convert_to_gpu_half  :error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err = clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    clReleaseMemObject(cl_data);
    free(buffer);
    t1=get_timestamp();
    FatherThreadTime = (t1 - t0) / 1000.0L;
    LOGI("DW_1_1_GPU Float To Half Time in %f ms\n", FatherThreadTime);

    return output;
}


//OpenCL conv
cnn_frame *doFeedForward_CONV_DW_GPU(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug


    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_dw_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    size_t localSize[3] = {8 , 8, 1};
//    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
//    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];
//    int didive = 8;
//    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c/2;
    cl_event  event;
    cl_int status;
    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);


    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_1_1_GPU(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug
    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);
    //debug


    int i = 0;
    cl_kernel kernel;
    if(((cnn_layer *)layer)->activation == RELU)
    {
        kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_1_1_kernel.kernel : openCLObjects->conv_kernel_float.kernel;
    }
    else  kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_1_1_kernel_no_relu.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = (output_c > 256 ? 256 : output_c);

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_3_3_GPU(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d \n",conv_layer->w,conv_layer->h,conv_layer->c);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug

    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_3_3_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    size_t localSize[3] = {8 , 8, 1};
//    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
//    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];
//    int didive = 8;
//    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c;

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, 0, 0
    );

    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_FIRST_GPU(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug


    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_first_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    size_t localSize[3] = {8 , 8, 1};
//    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
//    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];
//
//    int didive = 8;
//    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c;
    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_FC_GPU(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cl_int err = CL_SUCCESS;

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;
    OpenCLObjects *openCLObjects = getOpenClObject();

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug


    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_fc_kernel.kernel : openCLObjects->conv_fc_kernel_float.kernel;
    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    //size_t globalSize[1] = {(size_t)output->c};
    size_t globalSize[1] = {(size_t)(output->c % 256 == 0 ? 256 : output->c)};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            globalSize,
            0,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    frame_free(frame);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    return output;
}

//CPU
cnn_frame *doFeedForward_CONV(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = frame_convert_to_cpu(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    cnn_frame *output = frame_init(\
        (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1, \
        (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1, \
        conv_layer->n, ((cnn_layer *)layer)->useINT8);

    int i, j, k, x, y, z;
    for(i = 0 ; i < output->c; i++) {
        for(j = 0 ; j < output->h ; j++) {
            for(k = 0 ; k < output->w ; k++) {
                float result = 0.0f;
                for(x = 0 ; x < conv_layer->c; x++) {
                    for(y = 0 ; y < conv_layer->h; y++) {
                        for(z = 0 ; z < conv_layer->w ; z++) {
                            int w = k * conv_layer->stride[0] - conv_layer->pad[0] + z;
                            int h = j * conv_layer->stride[1] - conv_layer->pad[2] + y;
                            if(w < 0 || w >= frame->w)
                                continue;
                            if(h < 0 || h >= frame->h)
                                continue;

                            float tmp1 = getDataFrom3D(frame->data, frame->h, frame->w, frame->c, h, w, x);
                            float tmp2 = getDataFrom4D(conv_layer->W, conv_layer->n, conv_layer->h, conv_layer->w, conv_layer->c, i, y, z, x);
                            result += tmp1 * tmp2;
                        }
                    }
                }

                result += conv_layer->bias[i];
                output->data[getIndexFrom3D(output->c, output->h, output->w, i, j, k)] = result;
            }
        }
    }

    frame_free(frame);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    return output;
}

//Opencl conv INT8
//OPENCL image中才有INT8数据类型
cnn_frame *doFeedForward_CONV_DW_GPU_INT8(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

//    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);
//    if(((cnn_layer *)layer)->useHalf)
//        frame_convert_to_gpu_half(frame);
//    else if(((cnn_layer *)layer)->useINT8)
//        frame_convert_to_gpu_INT8(frame);
//    else frame_convert_to_gpu_float(frame);


    //转化数据暂时不管
    //数据准备完成
    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    int output_w = ((cnn_layer *)layer)->output_w;
    int output_h = ((cnn_layer *)layer)->output_h;
    int output_c = ((cnn_layer *)layer)->output_c;

    cnn_frame *output = frame_init_gpu_int8(output_w,output_h,output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug

    int i = 0;
//    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_dw_kernel.kernel : openCLObjects->conv_kernel_float.kernel;
    cl_kernel kernel = openCLObjects->conv_dw_INT8.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c/2;

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};
     cl_event event;
     cl_int status;
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("doFeedForward_CONV_DW_GPU_INT8  :error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_1_1_GPU_INT8(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    if(!frame->useGPU)
    {
        double t0,t1,time;
        t0 = get_timestamp();
        frame = frame_convert_to_gpu_int8(frame);
        t1 = get_timestamp();
        time = (t1 - t0) / 1000.0L;
        LOGI("data_pre time = %f ",time);
    }


    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

//    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);
    cnn_frame *output = frame_init_gpu_int8(output_w,output_h,output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug
    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);
    //debug


    int i = 0;
//    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_1_1_kernel.kernel : openCLObjects->conv_kernel_float.kernel;
    cl_kernel kernel = openCLObjects->conv_1_1_INT8.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    int global_x = (output_c > 256 ? 256 : output_c);
    int global_x = output_c/4;
    int global_y = output->h;
    int gs3 =  output_w;
//    int gs3 = output_c/16;
    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    cl_event event;
    cl_int status;
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("doFeedForward_CONV_1_1_GPU_INT8  :error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_3_3_GPU_INT8(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    if(!frame->useGPU)
    {
        double t0,t1,time;
        t0 = get_timestamp();
        frame = frame_convert_to_gpu_int8(frame);
        t1 = get_timestamp();
        time = (t1 - t0) / 1000.0L;
        LOGI("data_pre time = %f ",time);
    }


//    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

//    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);
    cnn_frame *output = frame_init_gpu_int8(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;


    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d \n",conv_layer->w,conv_layer->h,conv_layer->c);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug


    int i = 0;
//    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_3_3_kernel.kernel : openCLObjects->conv_kernel_float.kernel;
    cl_kernel kernel = openCLObjects->conv_3_3_INT8.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    size_t localSize[3] = {8 , 8, 1};
//    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
//    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];
//    int didive = 8;
//    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c;

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    cl_event event;
    cl_int status;
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, NULL, &event
    );

    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);


    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("doFeedForward_CONV_3_3_GPU_INT8  :error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_FIRST_GPU_INT8(cnn_frame *frame, void *layer) {
    LOGI("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

//    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    //int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_w = ((cnn_layer *)layer)->output_w;
    //int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_h = ((cnn_layer *)layer)->output_h;
    //int output_c = conv_layer->n;
    int output_c = ((cnn_layer *)layer)->output_c;

//    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);
    cnn_frame *output = frame_init_gpu_int8(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    //debug

    LOGI("output: W=%d H=%d C=%d \n",output_w,output_h,output_c);
    LOGI("kernel: w=%d h=%d c=%d n=%d \n",conv_layer->w,conv_layer->h,conv_layer->c,conv_layer->n);
    LOGI("input : w=%d h=%d c=%d \n",frame->w,frame->h,frame->c);

    //debug
/*//测试用
    cl_mem input_s = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            30 * sizeof(cl_char), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    cl_mem weight_s = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            30 * sizeof(cl_char), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);*/

    int i = 0;
//    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_first_kernel.kernel : openCLObjects->conv_kernel_float.kernel;
    cl_kernel kernel = openCLObjects->conv_first_INT8.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
//    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &input_s);
//    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &weight_s);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    size_t localSize[3] = {8 , 8, 1};
//    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
//    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];
//
//    int didive = 8;
//    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    int global_x = output->w;
    int global_y = output->h;
    int gs3 = output_c;
    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    cl_event event;
    cl_int status;
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            //localSize,
            NULL,
            0, NULL, &event
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);


    err=clWaitForEvents(1,&event);
    if(err!=CL_SUCCESS)
    {
        LOGI("doFeedForward_CONV_FIRST_GPU_INT8  :error event code : %d \n",err);
        err=clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL);
        if(status!=CL_COMPLETE);
        LOGI("error  here!!! %d\n",status);
    }

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);


/*//测试用
    char input_ss[30];
    char weight_ss[30];

    char *cl_input_ptr = (char *)clEnqueueMapBuffer(openCLObjects->queue, \
					input_s, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					30 * sizeof(char), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    char *cl_weight_ptr = (char *)clEnqueueMapBuffer(openCLObjects->queue, \
					weight_s, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					30 * sizeof(char), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    for (int j = 0; j < 30; ++j) {
        input_ss[j] = cl_input_ptr[j];
        weight_ss[j] = cl_weight_ptr[j];
    }

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					weight_s, \
					cl_weight_ptr, \
					0, NULL, NULL);//解除映射

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					input_s, \
					cl_input_ptr, \
					0, NULL, NULL);//解除映射*/
    frame_free(frame);

    return output;
}