#ifndef __BASIC_FUNCTIONS_HPP__
#define __BASIC_FUNCTIONS_HPP__

#include "deepsense_lib.hpp"

int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4);
float getDataFrom4D(float *data, int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4);
int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3);
float getDataFrom3D(float *data, int d1, int d2, int d3, int i1, int i2, int i3);

cnn_frame *activate_RAMP(cnn_frame *frame);
cnn_frame *activate_LOGISTIC(cnn_frame *frame);
cnn_frame *activate_RELU(cnn_frame *frame);
cnn_frame *activate_neon_RELU_INT8(cnn_frame *frame);
cnn_frame *activate_LEAKY(cnn_frame *frame);
cnn_frame *doFeedForward_Activation(cnn_frame *frame, int activation);

cnn_frame		*	frame_init(int w, int h, int c, int INT8);
cnn_frame       *   frame_init_not_align(int w, int h, int c, int INT8);
cnn_frame       *   frame_init_output(int w, int h, int c, int INT8);
cnn_frame       *   frame_init_share(int w, int h, int c);
cnn_frame 	    *	frame_init_gpu(int w, int h, int c);
cnn_frame       *   frame_init_gpu_half(int w, int h, int c);
cnn_frame       *   frame_init_gpu_int8(int w, int h, int c);
cnn_frame       *   frame_clone(cnn_frame *src);
cnn_frame       *   frame_convert_to_gpu_float(cnn_frame *frame);
cnn_frame       *   frame_convert_to_gpu_half(cnn_frame *frame);
cnn_frame       *   frame_convert_to_gpu_int8(cnn_frame *frame);
cnn_frame       *   frame_convert_to_cpu(cnn_frame *frame);
cnn_frame       *   frame_convert_to_cpu_int8(cnn_frame *frame);
cnn_frame       *   frame_cpu_pad(cnn_frame *frame);
cnn_frame       *   frame_convert_to_cpu_not_align(cnn_frame *frame);
void				frame_free(cnn_frame *frame);
void                frame_free_not_align(cnn_frame *frame);

//创建字节对齐函数
#define MALLOC_ALIGN    16
template<typename _Tp> _Tp* alignPtr(_Tp* ptr, int n);
int alignSize(int sz,int n);
void *fastMalloc(int dim,int w,int h,int c,int elemsize);
void fastFree(void *ptr);

//创建共享内存函数
int create_shared_memory(const char *name,int size,unsigned char *&addr,int &shm_id);
int open_shared_memory(unsigned char*& addr, int & shm_id);
int close_shared_memory(int & shm_id, unsigned char*& addr);


//socket通信
int SocketSendALL(char *buffer,int size);
int SocketRecvALL(char *buffer,int size);
void SocketServerCreate();
void SocketClientCreate();
void SocketServerClose();
void SocketClientClose();


//INT8量化
signed char float2int8(float v);
signed char* quantize_arm(float *input,signed char *output_s,int c,int h,int w,float scale,int aligned);
signed char* requantize_arm(int *input,int c,int h,int w,float *_bias,float scale_1,float scale_2,float *weight_scale);
float * dequantize_arm(int *input,int c,int h,int w,float *_bias,float layer_scale,float *weight_scale);

//填充函数
void fill(int *input,int size,int v);

void fill_float(float *input,int size,float v);

double myexp(double x);

float InvSqrt(float x);

float mySqrt(float m);
#endif
