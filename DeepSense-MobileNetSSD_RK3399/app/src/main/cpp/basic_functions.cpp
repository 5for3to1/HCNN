#include <basic_functions.hpp>
#include <math.h>
#include <malloc.h>
#include <predefine.hpp>
#include <clio.hpp>
#include <string.h>
#include <deepsense_internal_lib.hpp>
#include <sys/time.h>
#include <android/log.h>
#include <deepsense_lib.hpp>
//neon
#include <arm_neon.h>

//std
#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <stdint.h>

//创建共享内存相关头文件
#include <fcntl.h>
#include <linux/ashmem.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/ipc.h>
#include <linux/shm.h>
#include <android/sharedmem.h>

//建立SOCKET相关文件
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>


//INT8量化中的四舍五入函数
#include <math.h>

#ifndef LOG_PRINT
#define DEBUG_TAG "NDK_SampleActivity"
#define LOG_TAG "hellojni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#endif

timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
    return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

float getDataFrom4D(float *data, int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
    int index = i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
    return data[index];
}

int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3) {
    return i1 * (d2 * d3) + i2 * d3 + i3;
}

float getDataFrom3D(float *data, int d1, int d2, int d3, int i1, int i2, int i3) {
    int index = i1 * (d2 * d3) + i2 * d3 + i3;
    return data[index];
}

cnn_frame *activate_RAMP(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] = x * (x > 0) + 0.1 * x;
    }
    return frame;
}

cnn_frame *activate_LOGISTIC(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] = 1./(1. + exp(-x));
    }
    return frame;
}

cnn_frame *activate_neon_RELU(cnn_frame *frame) {

    int size = frame->h*frame->w;
//    int in_plane_size = (frame->h*frame->w + 15) & -16;  // 取对齐
    int in_plane_size = alignSize(frame->w * frame->h * sizeof(float), 16) /sizeof(float);
#pragma omp parallel for
    for (int i = 0; i < frame->c; ++i)
    {

        int nn = size >> 2;
        int remain = size - (nn << 2);
//        float* ptr = frame->data + frame->w*frame->h*i;

        float* ptr = frame->data + in_plane_size * i;

        if (nn > 0)
        {
            asm volatile(
            "veor       q1, q0, q0          \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1 :128]  \n"
            "vmax.f32   q0, q0, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d1}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(ptr)     // %1
            : "0"(nn),
            "1"(ptr)
            : "cc", "memory", "q0", "q1"
            );
        }
        for (; remain>0; remain--)
        {
            *ptr = std::max(*ptr, 0.f);
            ptr++;
        }
    }
    return frame;
}

cnn_frame *activate_neon_RELU_INT8(cnn_frame *frame){

    int channels = frame->c;
    int size = frame->w * frame->h;
    int in_plane_size = (frame->h*frame->w + 15) & -16;
#pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        signed char* ptr = frame->data_s8 + in_plane_size * q;

        int nn = size >> 4;
        int remain = size - (nn << 4);

        if (nn > 0)
        {
            asm volatile(
            "veor       q1, q0, q0          \n"    //异或
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.s8    {d0-d1}, [%1 :128]  \n"
            "vmax.s8    q0, q0, q1          \n"
            "subs       %0, #1              \n"
            "vst1.s8    {d0-d1}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(ptr)     // %1
            : "0"(nn),
            "1"(ptr)
            : "cc", "memory", "q0", "q1"
            );
        }

        for (; remain>0; remain--)
        {
            if (*ptr < 0)
                *ptr = 0;

            ptr++;
        }
    }

    return 0;
}

cnn_frame *activate_RELU(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] =(x > 0) ? x : 0;
    }
    return frame;
}

cnn_frame *activate_LEAKY(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] =(x > 0) ? x : 0.1 * x;
    }

    return frame;
}

cnn_frame *doFeedForward_Activation(cnn_frame *frame, int activation) {

    if(activation == NO_ACTIVATION)
        return frame;

    if(!frame->useGPU) {

        switch(activation) {
            case LOGISTIC:
                activate_LOGISTIC(frame);
                break;
            case RAMP:
                activate_RAMP(frame);
                break;
            case LEAKY:
                activate_LEAKY(frame);
                break;
            case RELU:
//                activate_RELU(frame);
                activate_neon_RELU(frame);
                break;
        }
    } else {
        OpenCLObjects *openCLObjects = getOpenClObject();
        cl_int err = CL_SUCCESS;
        int i = 0;

        cl_kernel kernel = (frame->useHalf) ? openCLObjects->activation_kernel.kernel : openCLObjects->activation_kernel_float.kernel;
        err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, i++, sizeof(int), &activation);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t globalSize[1] = {(size_t)(frame->w * frame->h * frame->c)};

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
    }

    return frame;
}

cnn_frame *frame_init(int w, int h, int c, int INT8) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 0;
    frame->useHalf = 0;
    frame->useINT8 = INT8;
    if(!INT8)
        frame->data = (float *)fastMalloc(3,w,h,c, sizeof(float));
    else
        frame->data_s8 = (signed char *)fastMalloc(3,w,h,c, sizeof(signed char));
    return frame;
}

cnn_frame *frame_init_not_align(int w, int h, int c, int INT8) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 0;
    frame->useHalf = 0;
    frame->useINT8 = INT8;
    if(!INT8)
        frame->data = (float *)malloc(w*h*c*sizeof(float));
    else
        frame->data_s8 = (signed char *)malloc(w*h*c*sizeof(signed char));
    return frame;
}

cnn_frame *frame_init_output(int w, int h, int c, int INT8) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 0;
    frame->useHalf = 0;
    frame->useINT8 = INT8;
    if(!INT8)
        frame->data = (float *)fastMalloc(3,w,h,c, sizeof(float));
    else
        frame->data_s8 = (signed char *)fastMalloc(3,w,h,c, sizeof(int));   //INT8 NEON输出帧
    return frame;
}

cnn_frame *frame_init_share(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->data =NULL;
    frame->useGPU = 0;
    frame->useHalf = 0;
    return frame;
}

cnn_frame *frame_init_gpu(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 1;
    frame->useHalf = 0;
    frame->useINT8 = 0;
    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    frame->cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            frame->w * frame->h * frame->c * sizeof(float), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(err == CL_SUCCESS)
        return frame;
    else {
        free(frame);
        return NULL;
    }
}

cnn_frame *frame_init_gpu_half(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 1;
    frame->useHalf = 1;
    frame->useINT8 = 0;

    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    frame->cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            frame->w * frame->h * frame->c * sizeof(cl_half), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(err == CL_SUCCESS)
        return frame;
    else {
        free(frame);
        return NULL;
    }
}

cnn_frame *frame_init_gpu_int8(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 1;
    frame->useHalf = 0;
    frame->useINT8 = 1;
    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    frame->cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            frame->w * frame->h * frame->c * sizeof(cl_char), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(err == CL_SUCCESS)
        return frame;
    else {
        free(frame);
        LOGI("init_gpu_int8 error!!!\n");
        return NULL;
    }
}

cnn_frame *frame_clone(cnn_frame *src) {
    if(!src->useGPU) {
        cnn_frame *frame;
        if(!src->useINT8)
        {
            frame = frame_init(src->w, src->h, src->c,src->useINT8);
            for (int i = 0; i < frame->c; ++i) {
                memcpy((void *) (frame->data + i * alignSize(frame->w * frame->h * sizeof(float), 16) / sizeof(float)), (void *) (src->data + i * alignSize(frame->w * frame->h * sizeof(float), 16) / sizeof(float)), frame->w * frame->h * sizeof(float));
            }
        }else{
            frame = frame_init_output(src->w, src->h, src->c,src->useINT8);
            int  *input =  (int*)frame->data_s8;
            int  *output = (int *)src->data_s8;
            for (int i = 0; i < frame->c; ++i) {
                memcpy((void *) (input + i * alignSize(frame->w * frame->h * sizeof(int), 16) / sizeof(int)), (void *) (output + i * alignSize(frame->w * frame->h * sizeof(int), 16) / sizeof(int)), frame->w * frame->h * sizeof(int));
            }
        }
        return frame;
    } else  //useGPU
    {
        cl_int err = CL_SUCCESS;
        cnn_frame *frame = NULL;

        if(src->useINT8)//useGPU useINT8
        {
            frame = frame_init_gpu_int8(src->w, src->h, src->c);
            if (frame == NULL)
                return NULL;

            int mapped_size = src->w * src->h * src->c * sizeof(cl_char);

            OpenCLObjects *openCLObjects = getOpenClObject();
            float *buf_src = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    src->cl_data, \
                    CL_TRUE, CL_MAP_READ, \
                    0, \
                    mapped_size, \
                    0, NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                LOGI("frame_clone failure");
            }
            SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

            float *buf_dst = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    frame->cl_data, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
                    mapped_size, \
                    0, NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                LOGI("frame_clone failure");
            }
            SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

            memcpy((void *) buf_dst, (void *) buf_src, mapped_size);

            clEnqueueUnmapMemObject(openCLObjects->queue, \
                    src->cl_data, \
                    buf_src, \
                    0, NULL, NULL);

            clEnqueueUnmapMemObject(openCLObjects->queue, \
                    frame->cl_data, \
                    buf_dst, \
                    0, NULL, NULL);
            return frame;
        }
        else//useGPU !useINT8
        {
            if (src->useHalf == 0)
                frame = frame_init_gpu(src->w, src->h, src->c);
            else
                frame = frame_init_gpu_half(src->w, src->h, src->c);

            if (frame == NULL)
                return NULL;

            int mapped_size = src->w * src->h * src->c * sizeof(cl_float);
            if (src->useHalf == 1)
                mapped_size = src->w * src->h * src->c * sizeof(cl_half);

            OpenCLObjects *openCLObjects = getOpenClObject();
            float *buf_src = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    src->cl_data, \
                    CL_TRUE, CL_MAP_READ, \
                    0, \
                    mapped_size, \
                    0, NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                LOGI("frame_clone failure");
            }
            SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

            float *buf_dst = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    frame->cl_data, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
                    mapped_size, \
                    0, NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                LOGI("frame_clone failure");
            }
            SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

            memcpy((void *) buf_dst, (void *) buf_src, mapped_size);

            clEnqueueUnmapMemObject(openCLObjects->queue, \
                    src->cl_data, \
                    buf_src, \
                    0, NULL, NULL);

            clEnqueueUnmapMemObject(openCLObjects->queue, \
                    frame->cl_data, \
                    buf_dst, \
                    0, NULL, NULL);
            return frame;
        }
    }
}

cnn_frame* frame_convert_to_gpu_float(cnn_frame *frame) {
    if(frame->useGPU && !frame->useHalf)
    {
        return frame;
    }

    OpenCLObjects *openCLObjects = getOpenClObject();
    cnn_frame *output = frame_init_gpu(frame->w, frame->h, frame->c);
    int err = CL_SUCCESS;

    if(!frame->useGPU) {
        //CPU-mode
        float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					output->cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					output->w * output->h * output->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//        memcpy((void *)buf_dest, frame->data, output->w * output->h * output->c * sizeof(cl_float));
        for (int i = 0; i < frame->c; ++i) {
            memcpy((void*)(buf_dest + i*frame->w*frame->h),
                   (void*)(frame->data + i*alignSize(frame->w*frame->h* sizeof(float),16)/ sizeof(float)), frame->w * frame->h * sizeof(float));
        }

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					output->cl_data, \
					buf_dest, \
					0, NULL, NULL);
    } else {
        //GPU-half-mode
        cl_kernel kernel = openCLObjects->convert_half_to_float_kernel.kernel;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t convertSize[1] = {(size_t) output->w * output->h * output->c};
        err = clEnqueueNDRangeKernel(
                openCLObjects->queue,
                kernel,
                1,
                0,
                convertSize,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    }

    frame_free(frame);

//    test
//    {
//        float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
//					output->cl_data, \
//					CL_TRUE, CL_MAP_READ, \
//					0, \
//					output->w * output->h * output->c * sizeof(cl_float), \
//					0, NULL, NULL, &err);
//        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
//
//        LOGI("first channel \n");
//        for (int j = 0; j < 32; ++j) {
//            LOGI("%f ",buf_dest[j]);
//        }
//
//        LOGI("second channel \n");
//        for (int j = 0; j < 32; ++j) {
//            LOGI("%f ",buf_dest[32+j]);
//        }
//
//        clEnqueueUnmapMemObject(openCLObjects->queue, \
//					output->cl_data, \
//					buf_dest, \
//					0, NULL, NULL);
//    }

    return output;
}

cnn_frame* frame_convert_to_gpu_half(cnn_frame *frame) {

    if(frame->useGPU && frame->useHalf)
        return frame;

    cnn_frame *output = frame_init_gpu_half(frame->w, frame->h, frame->c);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    cl_mem cl_data = NULL;

    if(!frame->useGPU) {
        //cpu-mode
        cl_data = clCreateBuffer(
                openCLObjects->context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                frame->w * frame->h * frame->c * sizeof(cl_float),
                NULL,
                &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        float *cl_data_ptr = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        for (int i = 0; i < frame->c; ++i) {
            memcpy((void*)(cl_data_ptr + i*frame->w*frame->h),
                   (void*)(frame->data + i*alignSize(frame->w*frame->h* sizeof(float),16)/ sizeof(float)), frame->w * frame->h * sizeof(float));
        }

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					cl_data, \
					cl_data_ptr, \
					0, NULL, NULL);
    } else
    {
        //gpu-float-mode
        cl_data = frame->cl_data;
    }

    cl_kernel kernel = openCLObjects->convert_float_to_half_kernel.kernel;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->cl_data);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t convertSize[1] = {(size_t) output->w * output->h * output->c};
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            convertSize,
            0,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err = clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(!frame->useGPU)
        clReleaseMemObject(cl_data);

    frame_free(frame);

    return output;
}

cnn_frame* frame_convert_to_gpu_int8(cnn_frame *frame) {

    cnn_frame *output = frame_init_gpu_int8(frame->w, frame->h, frame->c);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    //cpu-mode
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    signed char *cl_data_ptr = (signed char *)clEnqueueMapBuffer(openCLObjects->queue, \
					output->cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_char), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    int input_size = alignSize(frame->w*frame->h* sizeof(signed char),16)/ sizeof(signed char);

#pragma omp parallel for
    for (int i = 0; i < frame->c; ++i) {
        for (int j = 0; j < frame->w*frame->h; ++j) {
            cl_data_ptr[j*frame->c +i] = frame->data_s8[i*input_size + j];
        }
    }

     clEnqueueUnmapMemObject(openCLObjects->queue, \
					output->cl_data, \
					cl_data_ptr, \
					0, NULL, NULL);

    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    LOGI("frame_convert_to_gpu_int8 \n");
    frame_free(frame);

    return output;
}

cnn_frame * frame_convert_to_cpu(cnn_frame *frame) {
    if(!frame->useGPU)
        return frame;

    cnn_frame *output = frame_init(frame->w, frame->h, frame->c, frame->useINT8);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    //convert half to float first
    if(frame->useHalf) {

        cnn_frame *tmp = frame_init_gpu(frame->w, frame->h, frame->c);

        cl_kernel kernel = openCLObjects->convert_half_to_float_kernel.kernel;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t convertSize[1] = {(size_t) tmp->w * tmp->h * tmp->c};
        err = clEnqueueNDRangeKernel(
                openCLObjects->queue,
                kernel,
                1,
                0,
                convertSize,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        frame_free(frame);
        frame = tmp;
    }

    //map gpu-mem to cpu-mem and copy data
    float *buf_src = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

//    memcpy((void*)output->data, (void*)buf_src, frame->w * frame->h * frame->c * sizeof(cl_float));
    for (int i = 0; i < frame->c; ++i) {
        memcpy((void*)(output->data + i*alignSize(frame->w*frame->h* sizeof(float),16)/ sizeof(float)),
               (void*)(buf_src + i*frame->w*frame->h), frame->w * frame->h * sizeof(cl_float));
    }

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					buf_src, \
					0, NULL, NULL);

    frame_free(frame);

    return output;
}

cnn_frame* frame_convert_to_cpu_int8(cnn_frame *frame) {

    if(!frame->useGPU)
    {
        return frame;
    }

    cnn_frame *output = frame_init(frame->w, frame->h, frame->c,1);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    //cpu-mode
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    signed char *cl_data_ptr = (signed char *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_char), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    int output_size = alignSize(frame->w*frame->h* sizeof(signed char),16)/ sizeof(signed char);

#pragma omp parallel for
    for (int i = 0; i < frame->c; ++i) {
        for (int j = 0; j < frame->w*frame->h; ++j) {
            output->data_s8[i*output_size + j] = cl_data_ptr[j*frame->c +i];
        }
    }

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					cl_data_ptr, \
					0, NULL, NULL);

    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    LOGI("frame_convert_to_cpu_int8 \n");
    frame_free(frame);

    return output;
}

cnn_frame * frame_cpu_pad(cnn_frame *frame){

    cnn_frame *pad_frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    pad_frame->w = frame->w+2;
    pad_frame->h = frame->h+2;
    pad_frame->c = frame->c;
    pad_frame->useGPU = frame->useGPU;
    pad_frame->useHalf = frame->useHalf;
    pad_frame->useINT8 = frame->useINT8;

    if(!pad_frame->useINT8) {
        pad_frame->data = (float *) fastMalloc(3, pad_frame->w, pad_frame->h, pad_frame->c, sizeof(float));
        int align_size = alignSize(frame->w * frame->h * sizeof(float), 16) / sizeof(float);
        int pad_align_size = alignSize(pad_frame->w * pad_frame->h * sizeof(float), 16) / sizeof(float);
#pragma omp parallel for
        for (int i = 0; i < pad_frame->c; ++i) {
            memset((void *) (pad_frame->data + i * pad_align_size), 0, pad_frame->w * sizeof(float));
            for (int j = 1; j < pad_frame->h - 1; ++j) {
                pad_frame->data[i * pad_align_size + j * pad_frame->w] = 0;
                pad_frame->data[i * pad_align_size + j * pad_frame->w + pad_frame->w - 1] = 0;
                memcpy((void *) (pad_frame->data + i * pad_align_size + j * pad_frame->w + 1), frame->data + i * align_size + (j - 1) * frame->w, frame->w * sizeof(float));
            }
            memset((void *) (pad_frame->data + i * pad_align_size + (pad_frame->h - 1) * pad_frame->w), 0, pad_frame->w * sizeof(float));
        }
    }else{
        pad_frame->data_s8 = (signed char *) fastMalloc(3, pad_frame->w, pad_frame->h, pad_frame->c, sizeof(signed char));
        int align_size = alignSize(frame->w * frame->h * sizeof(signed char), 16) / sizeof(signed char);
        int pad_align_size = alignSize(pad_frame->w * pad_frame->h * sizeof(signed char), 16) / sizeof(signed char);

        for (int i = 0; i < pad_frame->c; ++i) {
            memset((void *) (pad_frame->data_s8 + i * pad_align_size), 0, pad_frame->w * sizeof(signed char));
#pragma omp parallel for
            for (int j = 1; j < pad_frame->h - 1; ++j) {
                pad_frame->data_s8[i * pad_align_size + j * pad_frame->w] = 0;
                pad_frame->data_s8[i * pad_align_size + j * pad_frame->w + pad_frame->w - 1] = 0;
                memcpy((void *) (pad_frame->data_s8 + i * pad_align_size + j * pad_frame->w + 1), frame->data_s8 + i * align_size + (j - 1) * frame->w, frame->w * sizeof(signed char));
            }
            memset((void *) (pad_frame->data_s8 + i * pad_align_size + (pad_frame->h - 1) * pad_frame->w), 0, pad_frame->w * sizeof(signed char));
        }

    }
    frame_free(frame);
    return pad_frame;
}

cnn_frame * frame_convert_to_cpu_not_align(cnn_frame *frame) {

    if(!frame->useGPU)
        return frame;

    cnn_frame *output = frame_init_not_align(frame->w, frame->h, frame->c, frame->useINT8);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    //convert half to float first
    if(frame->useHalf) {

        cnn_frame *tmp = frame_init_gpu(frame->w, frame->h, frame->c);

        cl_kernel kernel = openCLObjects->convert_half_to_float_kernel.kernel;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t convertSize[1] = {(size_t) tmp->w * tmp->h * tmp->c};
        err = clEnqueueNDRangeKernel(
                openCLObjects->queue,
                kernel,
                1,
                0,
                convertSize,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        frame_free(frame);
        frame = tmp;
    }

    //map gpu-mem to cpu-mem and copy data
    float *buf_src = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    memcpy((void*)output->data, (void*)buf_src, frame->w * frame->h * frame->c * sizeof(cl_float));

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					buf_src, \
					0, NULL, NULL);

    frame_free(frame);

    return output;
}

void frame_free(cnn_frame *frame) {
    if(frame->useGPU == 0)
    {
        if(frame->useINT8==0)  //这里free可以用useINT8来做条件
            fastFree(frame->data);
        else fastFree(frame->data_s8);
    }
    else {
        clReleaseMemObject(frame->cl_data);
    }
    free(frame);
}

void frame_free_not_align(cnn_frame *frame) {
    if(frame->useGPU == 0)
    {
        if(frame->data_s8==0)  //这里free可以用useINT8来做条件
            free(frame->data);
        else
            free(frame->data_s8);
    }
    else {
        clReleaseMemObject(frame->cl_data);
    }
    free(frame);
}


//创建指定大小维度的对齐的内存空间
//创建相关的所有函数
//创建内存字节对齐
template<typename _Tp> _Tp* alignPtr(_Tp* ptr, int n)
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

int alignSize(int sz,int n){
    return (sz+n-1)&-n;
}

//dim:创建空间的维度 elemsize：元素的大小
//c,h,w排列 返回void *adata指向创建的内存空间
void *fastMalloc(int dim,int w,int h,int c,int elemsize){
    unsigned char *udata;
    unsigned char **adata;
    int *refcount;
    int cstep;
    int total,totalsize;

    if(dim==1)
    {
        c=1;
        h=1;
        cstep=w;
        total=cstep*c;
        totalsize=alignSize(total*elemsize,4);
        udata = (unsigned char *) malloc(totalsize + (int) sizeof(*refcount)+sizeof(void *) + MALLOC_ALIGN);
        adata = alignPtr((unsigned char **) udata + 1, MALLOC_ALIGN);
        adata[-1] = udata;
        refcount = (int *) (((unsigned char *) adata) + totalsize);
        *refcount = 1;
    }else if(dim==2){
        c=1;
        cstep=w*h;
        total=cstep*c;
        totalsize=alignSize(total*elemsize,4);
        udata = (unsigned char *) malloc(totalsize + (int) sizeof(*refcount)+sizeof(void *) + MALLOC_ALIGN);
        adata = alignPtr((unsigned char **) udata + 1, MALLOC_ALIGN);
        adata[-1] = udata;
        refcount = (int *) (((unsigned char *) adata) + totalsize);
        *refcount = 1;
    }else if(dim==3) {
        cstep = alignSize(w*h*elemsize,16) / elemsize;
        total=cstep*c;
        totalsize=alignSize(total*elemsize,4);
        udata = (unsigned char *) malloc(totalsize + (int) sizeof(*refcount)+sizeof(void *) + MALLOC_ALIGN);
        adata = alignPtr((unsigned char **) udata + 1, MALLOC_ALIGN);
        adata[-1] = udata;
        refcount = (int *) (((unsigned char *) adata) + totalsize);
        *refcount = 1;
    }

    return adata;
}
//与fastFree配套使用

void fastFree(void *ptr){
    if(ptr){
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}
//创建字节对齐内存函数结束


//创建共享内存相关函数
#define ASHMEM_DEVICE "/dev/ashmem"
//创建内存大小的时候一定要注意是按字节多少来创建  !!!!
int create_shared_memory(const char *name,int size, unsigned char *&addr,int &shm_id){
    int fd = open(ASHMEM_DEVICE,O_RDWR);

    if(fd<0){
        return  -1;
    }
    int len=ioctl(fd,ASHMEM_GET_SIZE,NULL);
    if(len>0){
        addr = (unsigned char *)mmap(NULL,size,PROT_WRITE|PROT_READ,MAP_SHARED,fd,0);
        shm_id=fd;
        return 1;
    }else{
        int ret=ioctl(fd,ASHMEM_SET_NAME,name);
        if(ret<0){
            close(fd);
            return -1;
        }
        ret = ioctl(fd,ASHMEM_SET_SIZE,size);
        if(ret<0){
            close(fd);
            return -1;
        }
        addr =(unsigned char *)mmap(NULL,size,PROT_WRITE|PROT_READ,MAP_SHARED,fd,0);
        shm_id=fd;
    }
    return 0;
}

int open_shared_memory(unsigned char*& addr, int & shm_id){
    int size = ioctl(shm_id, ASHMEM_GET_SIZE,NULL);
    if(size > 0){
        addr = (unsigned char*)mmap(NULL, size , PROT_READ | PROT_WRITE, MAP_SHARED, shm_id, 0);
    }else{
        return -1;
    }
    return 0;
}

int close_shared_memory(int & shm_id, unsigned char*& addr){
    int size = ioctl(shm_id, ASHMEM_GET_SIZE, NULL);
    if(size <0){
        return -1;
    }
    int ret = munmap((void*)addr, size);
    if(ret == -1){
        return -1;
    }
    ret = close(shm_id);
    if(ret == -1){
        return -1;
    }
    return 0;
}
//创建共享内存结束


//创建socket通信相关函数
#define SERVER_PORT 9999
char server_ip[20]="127.0.0.1";     //这个东西不能放在头文件中，不然就报错了

//主进程与子进程TCP连接
extern int server_fd, client_fd;


//此处size也为传输的字节数
int SocketSendALL(char *buffer,int size){
    while(size>0){
        int SendSize=send(client_fd,buffer,size,0);
        if(SendSize<0)
        {
            LOGI("SOCKET SEND ERROR!!!\n");
            return -1;
        }
        size=size-SendSize;
        buffer+=SendSize;
    }
    return 0;
}

int SocketRecvALL(char *buffer,int size){
    while(size>0){
        int RecvSize=recv(client_fd,buffer,size,0);
        if(RecvSize<0)
        {
            LOGI("SOCKET RECEVE ERROR!!!\n");
            return -1;
        }
        size=size-RecvSize;
        buffer+=RecvSize;
    }
    return 0;
}

//返回连接上的客户端的嵌套字
void SocketServerCreate(){

    sockaddr_in server_addr;

    memset(&server_addr,0, sizeof(server_addr));   //初始化网络地址

    server_addr.sin_family=AF_INET;                 //设置服务端网络地址-协议簇（sin_family）

    server_addr.sin_addr.s_addr=INADDR_ANY;         //设置服务端IP地址（自动获取系统默认的本机IP，自动分配）

    server_addr.sin_port=htons(SERVER_PORT);       //设置服务端端口 9999

    server_fd=socket(PF_INET,SOCK_STREAM,0);//创建服务端SOCKET 通信域（IPV4） 通信类型（SOCKET流） 使用协议（一般为0）

    if(server_fd<0){
        LOGI("CREATE SERVER SOCKET ERROR!!!\n");
    }

    LOGI("CREATE SERVER SOCKET SUCCESS!!!\n");

    int optval=1;
    if(0 > setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&optval,sizeof(optval)))
    {
        LOGI("Failed to set address reuse!");
    }

    bind(server_fd,(sockaddr *)&server_addr, sizeof(server_addr));  //服务端绑定地址

    listen(server_fd,6);                    //监听客户端连接请求   监听的服务端socket  客户端数量

    sockaddr_in client_addr;
    socklen_t  sin_size= sizeof(sockaddr_in);
    client_fd=accept(server_fd,(sockaddr*)&client_addr,&sin_size);  //接收客户端连接

    if(client_fd<0){
        LOGI("ACCEPT CLIENT ERROR!!!\n");
    }

    LOGI("ACCEPT CLIENT SUCCESS!!!\n");
}
//返回客户端的嵌套字
void SocketClientCreate(){

    sockaddr_in server_addr;

    memset(&server_addr,0, sizeof(server_addr));

    server_addr.sin_family=AF_INET;

    server_addr.sin_addr.s_addr=inet_addr(server_ip);

    server_addr.sin_port=htons(SERVER_PORT);

    client_fd=socket(PF_INET,SOCK_STREAM,0);

    if(client_fd<0){
        LOGI("CREATE CLIENT SOCKET ERROR!!!\n");
    }

    int con_result=connect(client_fd,(sockaddr*)&server_addr,sizeof(server_addr));

    if(con_result<0){
        LOGI("CLINET CONNECT ERROR!!!\n");
    }
    LOGI("CLINET CONNECT SUCCESS!!!\n");
}

//服务器端断开TCP连接
void SocketServerClose()
{
    close(client_fd);
    close(server_fd);
}
//客户端断开TCP连接
void SocketClientClose()
{
    close(client_fd);
}


//INT8量化 在实际调用中需要考虑量化比例再传入参数量化
signed char float2int8(float v)
{
    int int32 = round(v);   //四舍五入函数
    if (int32 > 127) return 127;
    if (int32 < -128) return -128;
    return (signed char)int32;
}

//此处为flaot->int8 考虑了是否为对齐的内存空间和是否指定输出内存空间
signed char* quantize_arm(float *input,signed char *output_s,int c,int h,int w,float scale,int aligned){
    int size = w*h;
    int in_plane_size;
    int out_plane_size;
    signed char *output;

    if (aligned) {
        in_plane_size = alignSize(size * sizeof(float), 16) /sizeof(float);
        out_plane_size= alignSize(size * sizeof(signed char), 16) /sizeof(signed char);
        if(output_s==NULL)
            output = (signed char *) fastMalloc(3, w, h, c, sizeof(signed char));
        else output = output_s;
    } else {
        in_plane_size = h * w;
        out_plane_size = h*w;
        if(output_s==NULL)
            output = (signed char *) malloc(w * h * c * sizeof(signed char));
        else output = output_s;
    }

#pragma omp parallel for
    for (int q=0; q<c; q++)
    {
        const float* ptr = input + q*in_plane_size;
        signed char* outptr = output + q*out_plane_size;

        int nn = size >> 3;
        int remain = size & 7;

        if (nn > 0)
        {
            asm volatile(
            "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"
                    "vdup.32    q10, %6             \n"

                    "0:                             \n"
                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vcvtr.s32.f32 s0,s0            \n"
                    "vcvtr.s32.f32 s1,s1            \n"
                    "vcvtr.s32.f32 s2,s2            \n"
                    "vcvtr.s32.f32 s3,s3            \n"
                    "vcvtr.s32.f32 s4,s4            \n"
                    "vcvtr.s32.f32 s5,s5            \n"
                    "vcvtr.s32.f32 s6,s6            \n"
                    "vcvtr.s32.f32 s7,s7            \n"

                    "vqmovn.s32 d4,q0               \n"  //向量饱和窄移 宽度缩小一半
                    "vqmovn.s32 d5,q1               \n"

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"

                    "vqmovn.s16 d4, q2              \n" //向量饱和窄移 宽度缩小一半
                    "vst1.8     {d4}, [%2]!         \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
            : "=r"(nn),         // %0
            "=r"(ptr),        // %1
            "=r"(outptr)      // %2
            : "0"(nn),
            "1"(ptr),
            "2"(outptr),
            "r"(scale)        // %6
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11"
            );
        }

        for (; remain>0; remain--)
        {
            *outptr = float2int8(*ptr * scale);

            ptr++;
            outptr++;
        }
    }

    if(aligned==1)
        fastFree(input);
    else
        free(input);

    return output;
}

//此处专为数据帧的重量化  针对对齐的数据格式 dim_term==0 dim==3
signed char* requantize_arm(int *input,int c,int h,int w,float *_bias,float scale_1,float scale_2,float *weight_scale){
    int size = w*h;
    int in_plane_size = alignSize(size * sizeof(int), 16) /sizeof(int);
    int out_plane_size = alignSize(size * sizeof(signed char), 16) /sizeof(signed char);
    signed char *output = (signed char*)fastMalloc(3,w,h,c,sizeof(signed char));
    float scale_out = scale_2;

#pragma omp parallel for
    for (int q=0; q<c; q++)
    {
        const int* intptr = input + in_plane_size*q;
        signed char* ptr = output + out_plane_size*q;
        float scale_in = 1/(scale_1 * weight_scale[q]);
        float bias = _bias[q];

        int nn = size >> 3;
        int remain = size & 7;

        if (nn > 0)
        {
            asm volatile(
            "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale_in
                    "vdup.f32   q11, %7             \n" //q11 scale_out
                    "vdup.f32   q12, %8             \n" //q12 bias
                    "0:                             \n"
                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q0, q0            \n"
                    "vcvt.f32.s32 q1, q1            \n"
                    // top_f32 = top_f32 * scale_int
                    "vmul.f32   q0, q0, q10         \n"
                    "vmul.f32   q1, q1, q10         \n"
                    // top_f32 = top_f32 + bias
                    "vadd.f32   q0, q0, q12         \n"
                    "vadd.f32   q1, q1, q12         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q11         \n"
                    "vmul.f32   q1, q1, q11         \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s4, s4           \n"
                    "vcvtr.s32.f32 s5, s5           \n"
                    "vcvtr.s32.f32 s6, s6           \n"
                    "vcvtr.s32.f32 s7, s7           \n"
                    // top_s32 -> top_s16
                    "vqmovn.s32 d4, q0              \n"
                    "vqmovn.s32 d5, q1              \n"
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1:128]!  \n" //q0-q1 data
                    // top_s16 -> top_s8
                    "vqmovn.s16   d4, q2            \n"
                    // save top_s8
                    "vst1.8     {d4}, [%2:64]!      \n"
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    "sub        %1, #32             \n"
            : "=r"(nn),         // %0
            "=r"(intptr),     // %1
            "=r"(ptr)         // %2
            : "0"(nn),
            "1"(intptr),
            "2"(ptr),
            "r"(scale_in),    // %6
            "r"(scale_out),   // %7
            "r"(bias)         // %8
            : "cc", "memory", "q0", "q1", "q2", "q10", "q11", "q12"
            );
        }

        for (; remain > 0; remain--)
        {
            *ptr = float2int8(((*intptr * scale_in) + bias) * scale_out);

            intptr++;
            ptr ++;
        }
    }

    fastFree(input);
    return  output;
}

//反量化，用在网络的输出帧转化  兼容全精度的detection_output创建对齐空间
float * dequantize_arm(int *input,int c,int h,int w,float *_bias,float layer_scale,float *weight_scale){
    int size = w*h;
    int in_plane_size = alignSize(size * sizeof(int), 16) /sizeof(int);
    int out_plane_size = alignSize(size * sizeof(float), 16) /sizeof(float);
    float *output = (float*)fastMalloc(3,w,h,c,sizeof(float));

#pragma omp parallel for
    for (int q=0; q<c; q++)
    {
        int* intptr = input + in_plane_size*q;
        float* ptr = output + out_plane_size*q;

        float bias = _bias[q];
        float scale = 1/(layer_scale * weight_scale[q]);

        int nn = size >> 3;
        int remain = size & 7;

        if (nn > 0)
        {
            asm volatile(
            "pld        [%1, #256]          \n" //intptr 8个int32
                    "vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale
                    "vdup.f32   q12, %7             \n" //q12 bias

                    "0:                             \n"
                    "vcvt.f32.s32 q0, q0            \n"
                    "vcvt.f32.s32 q1, q1            \n"

                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vadd.f32   q2,q0,q12           \n"
                    "vadd.f32   q3,q1,q12           \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n"  //intptr
                    "vst1.f32   {d4-d7}, [%2]!      \n"  //ptr

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
            : "=r"(nn),         // %0
            "=r"(intptr),     // %1
            "=r"(ptr)         // %2
            : "0"(nn),
            "1"(intptr),
            "2"(ptr),
            "r"(scale),       // %6
            "r"(bias)         // %7
            : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12"
            );
        }

        for (; remain>0; remain--)
        {
            *ptr = *intptr * scale + bias;

            intptr++;
            ptr++;
        }
    }

    fastFree(input);
    return  output;

}

//填充函数  填充int结构
void fill(int *input,int size,int v){
    int* ptr = input;

    int nn = size >> 2;
    int remain = size - (nn << 2);

    int32x4_t _c = vdupq_n_s32(v);

    if (nn > 0)
    {
        asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.s32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
        "=r"(ptr)     // %1
        : "0"(nn),
        "1"(ptr),
        "w"(_c)       // %4
        : "cc", "memory"
        );
    }

    for (; remain>0; remain--)
    {
        *ptr++ = v;
    }

}

//填充函数，填充float结构
void fill_float(float *input,int size,float v){

    float* ptr = input;

    int nn = size >> 2;
    int remain = size - (nn << 2);

    float32x4_t _c = vdupq_n_f32(v);

    if (nn > 0)
    {
        asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
        "=r"(ptr)     // %1
        : "0"(nn),
        "1"(ptr),
        "w"(_c)       // %4
        : "cc", "memory"
        );
    }

    for (; remain>0; remain--)
    {
        *ptr++ = v;
    }


}


double pow_i(double num,int n)//计算num的n次幂，其中n为整数
{
    double powint=1;
    int i;
    for(i=1;i<=n;i++) powint*=num;
    return powint;
}

double pow_f(double num,double m)//计算num的m次幂，num和m可为双精度，num大于零
{
    int i,j;
    double powf=0,x,tmpm=1;
    x=num-1;
    for(i=1;tmpm>1e-12 || tmpm<-1e-12;i++)//当tmpm不在次范围时，停止循环,范围可改
    {
        for(j=1,tmpm=1;j<=i;j++)
            tmpm*=(m-j+1)*x/j;
        powf+=tmpm;
    }
    return powf+1;
}

double pow_ff(double num,double m)//调用pow_f()和pow_i(),计算num的m次幂,是计算幂的入口
{
    if(num==0 && m!=0) return 0;//若num为0，则返回0
    else if(num==0 && m==0) return 1;// 若num和m都为0，则返回1
    else if(num<0 && m-int(m)!=0) return 0;//若num为负，且m不为整数数，则出错，返回0
    if(num>2)//把底数大于2的情况转为(1/num)^-m计算
    {
        num=1/num;
        m=-m;
    }
    if(m<0) return 1/pow_ff(num,-m);//把指数小于0的情况转为1/num^-m计算
    if(m-int(m)==0) return pow_i(num,m);/*当指数为浮点数是，分成整数和小数分别求
	                                    幂，这是因为但底数较小式，用pow_f直接求幂
										误差大，所以分为指数的整数部分用pow_i,小
										数部分用pow_f求.*/
    else return pow_f(num,m-int(m))*pow_i(num,int(m));
    return pow_f(num,m);
}

double mypow(double x,unsigned n)
{
    /*简化版乘方*/
    double s = 1;

    int i = n;
    while(i > 0)
    {
        s*=x;
        i--;
    }
    return s;
}

double myexp(double x){
    int i,k,m,t;
    int xm=(int)x;
    double sum;
    double e ;
    double ef;
    double z ;
    double sub=x-xm;
    m=1;      //阶乘算法分母
    e=1.0;  //e的xm
    ef=1.0;
    t=10;      //算法精度
    z=1;  //分子初始化
    sum=1;
    if (xm<0) {     //判断xm是否大于0？
        xm=(-xm);
        for(k=0;k<xm;k++){ef*=2.718281;}
        e/=ef;
    }
    else { for(k=0;k<xm;k++){e*=2.718281;} }

    for(i=1;i<t;i++){
        m*=i;
        z*=sub;
        sum+=z/m;
    }
    return sum*e;
}

float InvSqrt(float x) {
    float xhalf = 0.5f*x;
    int i = *(int*)&x; // get bits for floating VALUE
    i = 0x5f375a86- (i>>1); // gives initial guess y0
    x = *(float*)&i; // convert bits BACK to float
    x = x*(1.5f-xhalf*x*x); // Newton step, repeating increases accuracy
    return x;
}

float mySqrt(float m) {
    float i=0;
    float x1,x2;
    while ((i*i)<=m)
    {
        i+=0.1;
    }
    x1=i;
    for (int j=0;j<10;j++)
    {
        x2=m;
        x2/=x1;
        x2+=x1;
        x2/=2;
        x1=x2;
    }
    return 1/x2;
}