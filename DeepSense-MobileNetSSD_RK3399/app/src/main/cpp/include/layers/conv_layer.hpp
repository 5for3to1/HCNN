#ifndef __CONV_LAYER__
#define __CONV_LAYER__

#include <deepsense_lib.hpp>

//neon conv
cnn_frame *doFeedForward_CONV_1_1_NEON(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_S1_NEON(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_S2_NEON(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_3_3_S2_NEON(cnn_frame *frame, void *layer);

//neon int8
cnn_frame *doFeedForward_CONV_1_1_NEON_S8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_1_1_NEON_S8_LEFT4(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_1_1_NEON_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_3_3_S2_NEON_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_S1_NEON_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_S2_NEON_INT8(cnn_frame *frame, void *layer);

//mix conv
void doFeedForward_CONV_ASM_NEON_MIX(cnn_frame *frame, void *layer,float *shm2_buffer);
cnn_frame *doFeedForward_CONV_DW_1_1_MIX(cnn_frame *frame, void *layer,float *shm2_buffer,int client_fd,int LaterNum);
cnn_frame *doFeedForward_CONV_DW_MIX(cnn_frame *frame, void *layer,float *shm1_buffer,int client_fd,int LaterNum);

//OpenCL conv
cnn_frame *doFeedForward_CONV_FIRST_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_DW_1_1_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_1_1_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_3_3_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_FC_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV(cnn_frame *frame, void *layer);

//OpenCL conv INT8
cnn_frame *doFeedForward_CONV_DW_GPU_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_1_1_GPU_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_3_3_GPU_INT8(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_FIRST_GPU_INT8(cnn_frame *frame, void *layer);


//OpenCL Test
int TestOpencl();
#endif
