#ifndef __PREDEFINE_H
#define __PREDEFINE_H

#include <CL/cl.h>

#define PROGRAM_NAME "DEEPSENSE"
#define PROGRAM_KERNEL_NAME "deepsense.cl"

typedef struct {
    cl_kernel kernel;
    size_t kernel_max_workgroup_size;
} lany_kernel;

struct OpenCLObjects {
    // Regular OpenCL objects:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;     //we use single queue only
    cl_program program;

    //half kernels
    lany_kernel conv_first_kernel;
    lany_kernel conv_dw_kernel;
    lany_kernel conv_1_1_kernel;
    lany_kernel conv_1_1_kernel_no_relu;
    lany_kernel conv_3_3_kernel;
    lany_kernel conv_fc_kernel;
    lany_kernel activation_kernel;
    lany_kernel fully_connected_kernel;
    lany_kernel maxpool_kernel;
    lany_kernel lrn_kernel;

    lany_kernel convert_float_to_half_kernel_bias;
    lany_kernel convert_float_to_half_kernel;
    lany_kernel convert_half_to_float_kernel;

    //float kernels
    lany_kernel conv_kernel_float;
    lany_kernel conv_fc_kernel_float;
    lany_kernel fully_connected_kernel_float;
    lany_kernel maxpool_kernel_float;
    lany_kernel lrn_kernel_float;
    lany_kernel activation_kernel_float;

    //INT8 kernels
    lany_kernel conv_first_INT8;
    lany_kernel conv_dw_INT8;
    lany_kernel conv_1_1_INT8;
    lany_kernel conv_3_3_INT8;

    //test opencl

};

#endif
