#ifndef __LNN_LIB_HPP__
#define __LNN_LIB_HPP__

#include <CL/cl.h>

/*
 * This structure acts as both input and output to CNN Layer
 */
typedef struct {
	int w;
	int h;
	int c;
	float *data;
	signed char *data_s8;
	int useINT8;
	int useGPU;
	int useHalf;
	cl_mem cl_data;
} cnn_frame;

typedef enum {
	NO_ACTIVATION,
	RAMP,
	LOGISTIC,
	LEAKY,
	LINEAR,
	RELU
} activation_function;

typedef enum {\
	LAYER_TYPE_UNKNOWN = 0, \
	LAYER_TYPE_CONV, \
	LAYER_TYPE_FULLY_CONNECTED, \
	LAYER_TYPE_MAXPOOL, \
    LAYER_TYPE_SOFTMAX, \
    LAYER_TYPE_LRN_NORMALIZE, \
    LAYER_TYPE_LSTM
} layer_type;

typedef struct {
	int clip;
	int clip_count;
	int input_size;
	int output_size;
	float *W_x; //[4*output_size x input_size]
	float *W_h;
	float *bias;
	cl_mem cl_W_x;
	cl_mem cl_W_h;
	cl_mem cl_bias;
	//internal states
	float *prev_H;
	float *prev_C;
	cl_mem cl_prev_H;
	cl_mem cl_prev_C;
	int forward_temp_data;
	int need_reshape;
} cnn_layer_lstm;

typedef struct {
	int k;
	int size;
	float alpha;
	float beta;
} cnn_layer_lrn;

typedef struct {
	int stride[2];
	int pad[4];
	int w;	//width
	int h;	//height
	int c;	//channel
	int n;	//number of neurons
	int group;
	float *W;
	float *bias;
	signed char *W_S8;
	signed char *bias_s8;
	float *W_Scale;
	cl_mem cl_W;
	cl_mem cl_bias;
} cnn_layer_conv;

typedef struct {
	int weightSize;
	int inputSize;
	int outputSize;
	float *W;
	float *bias;
	cl_mem cl_W;
	cl_mem cl_bias;
	int need_reshape;
} cnn_layer_fully_connected;

typedef struct {
	int size;
	int stride[2];
	int pad[4];
} cnn_layer_maxpool;

typedef struct {
	int num_class;
	float nms_threshold;
	int nms_top_k;
	int keep_top_k;
	float confidence_threshold;
//	float variances[4];
//	int branch_num;
//	int * branch_anchor_num;
//	float *** anchor_size;
} cnn_layer_detection_output;

struct BBoxRect
{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int label;
};

typedef struct {
	int index;
	int mixCPUGPU;
	int useGPU;
	int useINT8;
	int useHalf;
	int num_threads;
	int output_w;
	int output_h;
	int output_c;
	float Scale_in;
	layer_type type;
	cnn_layer_conv *conv_layer;
	cnn_layer_detection_output *detection_output_layer;
	cnn_layer_fully_connected *connected_layer;
	cnn_layer_maxpool *maxpool_layer;
	cnn_layer_lrn *lrn_layer;
	cnn_layer_lstm *lstm_layer;
	cnn_frame *(*doFeedForward)(cnn_frame *frame, void *layer);
	cnn_frame *(*dwFeedForward)(cnn_frame *frame, void *layer,float *shm1_buffer,int client_fd,int LayerNum);
	cnn_frame *(*detectOutputForward)(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer);
	int activation;
} cnn_layer;

typedef struct {
	int nLayers;
	int useGPU;
	int useHalf;
	int useINT8;
	int input_w;
	int input_h;
	int input_c;
	float *averageImage;
	cnn_layer *layers;
} cnn;

cnn  *	cnn_loadModel(const char *modelDirPath, int useGPU);
void	cnn_free(cnn *model);

#endif
