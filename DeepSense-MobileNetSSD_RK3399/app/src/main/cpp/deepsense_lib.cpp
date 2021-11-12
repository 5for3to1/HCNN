//
// Created by JC1DA on 6/3/16.
//
#include <android/log.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <deepsense_lib.hpp>
#include <predefine.hpp>
#include <deepsense_internal_lib.hpp>
#include <clio.hpp>
#include <basic_functions.hpp>
#include <layers/conv_layer.hpp>
#include <layers/detection_output.hpp>
#include <layers/fully_connected.hpp>
#include <layers/maxpool.hpp>
#include <layers/softmax.hpp>
#include <layers/lrn.hpp>


#ifndef LOG_PRINT
#define DEBUG_TAG "NDK_SampleActivity"
#define LOG_TAG "hellojni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#endif

static inline int CMP_OPTION(char *str, const char *option) {
    int ret = strncmp(str, option, strlen(option)) == 0 ? 1 : 0;
    return ret;
}

static inline int PARSE_ACTIVATION(char *line) {
    char buf[32];
    sscanf(line,"ACTIVATION: %s\n",buf);
    if(CMP_OPTION(buf, "RAMP"))
        return RAMP;
    else if(CMP_OPTION(buf, "LOGISTIC"))
        return LOGISTIC;
    else if(CMP_OPTION(buf, "LEAKY"))
        return LEAKY;
    else if(CMP_OPTION(buf, "RELU"))
        return RELU;
    return NO_ACTIVATION;
}

cnn *cnn_loadModel(const char *modelDirPath, int useGPU) {

    LOGI("%s \n",modelDirPath);
    cnn *model = (cnn *)calloc(1, sizeof(cnn));
    model->useGPU = 0;
    model->useINT8 = 0;

    //读model文件
    {
        /* read number of layers */
        char fileNameBuf[256];
        char line[256];
        sprintf(fileNameBuf,"%s/model",modelDirPath);
        FILE *fp = fopen(fileNameBuf,"rb");
        if(fp == NULL)
        {
            LOGI("no model \n");
            return NULL;
        }
        LOGI("read model success \n");

        //PLEASE FILL IN NEW FORMAT
        while(fgets(line, sizeof(line), fp)) {
            if(CMP_OPTION(line, "NUMLAYERS"))
                sscanf(line, "NUMLAYERS: %d\n", &model->nLayers);
            else if(CMP_OPTION(line, "W"))
                sscanf(line, "W: %d\n", &model->input_w);
            else if(CMP_OPTION(line, "H"))
                sscanf(line, "H: %d\n", &model->input_h);
            else if(CMP_OPTION(line, "C"))
                sscanf(line, "C: %d\n", &model->input_c);
        }
        fclose(fp);
    }

    model->layers = (cnn_layer *)calloc(model->nLayers+1, sizeof(cnn_layer));
    cnn_layer *layers = model->layers;


    //在这里统一读取每层的blob_scale int8才用  在这里统一读取 不管是否
    if(model->useINT8)
    {
        char fileNameBuf[256];
        sprintf(fileNameBuf, "%s/layer_scale", modelDirPath);
        LOGI("%s \n", fileNameBuf);
        char LayerScaleFilePath[256];
        strcpy(LayerScaleFilePath, fileNameBuf);
        FILE *LayerScalefp = fopen(LayerScaleFilePath, "r");
        //有layer_scale文件
        if (LayerScalefp != NULL) {
            float *layer_scale_buffer = new float[model->nLayers];
            fread(layer_scale_buffer, sizeof(float), model->nLayers, LayerScalefp);

            for (int j = 0; j < model->nLayers; ++j) {
                cnn_layer *layer = &layers[j];
                layer->Scale_in = layer_scale_buffer[j];
            }
            fclose(LayerScalefp);
        }
    }


    int bottom[6]={22,26,28,30,32,34};

    for(int i = 1 ; i <= model->nLayers; i++) {
        char fileNameBuf[256];
        char line[256];

        sprintf(fileNameBuf,"%s/l_%d",modelDirPath,i);

        cnn_layer *layer = &layers[i - 1];
        layer->index = i - 1;

        layer->mixCPUGPU = 1;
        layer->useGPU = 0;
        layer->useINT8 = 0;

        layer->type = LAYER_TYPE_UNKNOWN;
        layer->activation = NO_ACTIVATION;

        LOGD("Loading layer %d\n", i);

        FILE *layerfp = fopen(fileNameBuf,"r");
        while (fgets(line, sizeof(line), layerfp)) {
            if(layer->type == LAYER_TYPE_UNKNOWN) {
                if(CMP_OPTION(line, "CONV")) {
                    layer->type = LAYER_TYPE_CONV;
                    layer->conv_layer = (cnn_layer_conv *) calloc(1, sizeof(cnn_layer_conv));

                    layer->conv_layer->group = 1;
                } else if(CMP_OPTION(line, "FULLY_CONNECTED")) {
                    layer->type = LAYER_TYPE_FULLY_CONNECTED;
                    layer->connected_layer = (cnn_layer_fully_connected *) calloc(1,
                                                                                  sizeof(cnn_layer_fully_connected));
                    layer->connected_layer->need_reshape = 0;
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_FULLY_CONNECTED;
                    else
                        layer->doFeedForward = doFeedForward_FULLY_CONNECTED_GPU;
                } else if(CMP_OPTION(line, "MAXPOOL")) {
                    layer->type = LAYER_TYPE_MAXPOOL;
                    layer->maxpool_layer = (cnn_layer_maxpool *) calloc(1,
                                                                        sizeof(cnn_layer_maxpool));
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_MAXPOOL;
                    else {
                        layer->doFeedForward = doFeedForward_MAXPOOL_GPU;
                    }
                } else if(CMP_OPTION(line, "SOFTMAX")) {
                    layer->type = LAYER_TYPE_SOFTMAX;
                    layer->doFeedForward = doFeedForward_SOFTMAX;
                } else if(CMP_OPTION(line, "LRN_NORM")) {
                    layer->type = LAYER_TYPE_LRN_NORMALIZE;
                    layer->lrn_layer = (cnn_layer_lrn *) malloc(sizeof(cnn_layer_lrn));
                    layer->lrn_layer->k = 1;
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_LRN;
                    else
                        layer->doFeedForward = doFeedForward_LRN_GPU;
                }
            }
            else
            {
                if(CMP_OPTION(line, "USE_HALF")) {
                    sscanf(line,"USE_HALF: %d", &layer->useHalf);
                    if(layer->useHalf != 0)
                        layer->useHalf = 1;
                }

                switch(layer->type) {
                    case LAYER_TYPE_CONV:
                        if(CMP_OPTION(line, "STRIDE")) {
                            sscanf(line, "STRIDE: %d %d\n", \
                                &layer->conv_layer->stride[0], \
                                &layer->conv_layer->stride[1]);
                        } else if(CMP_OPTION(line, "PAD")) {
                            sscanf(line,"PAD: %d %d %d %d\n", \
								&layer->conv_layer->pad[0], \
								&layer->conv_layer->pad[1], \
								&layer->conv_layer->pad[2], \
								&layer->conv_layer->pad[3]);
                        } else if(CMP_OPTION(line, "WIDTH")) {
                            sscanf(line,"WIDTH: %d\n",&layer->conv_layer->w);
                        } else if(CMP_OPTION(line, "HEIGHT")) {
                            sscanf(line,"HEIGHT: %d\n",&layer->conv_layer->h);
                        } else if(CMP_OPTION(line, "IN_CHANNELS")) {
                            sscanf(line,"IN_CHANNELS: %d\n",&layer->conv_layer->c);
                        } else if(CMP_OPTION(line, "OUT_CHANNELS")) {
                            sscanf(line,"OUT_CHANNELS: %d\n",&layer->conv_layer->n);
                        } else if(CMP_OPTION(line, "ACTIVATION")) {
                            layer->activation = PARSE_ACTIVATION(line);
                        } else if(CMP_OPTION(line, "GROUP")) {
                            sscanf(line,"GROUP: %d\n",&layer->conv_layer->group);
                        }
                        break;
                    case LAYER_TYPE_FULLY_CONNECTED:
                        if(CMP_OPTION(line, "INPUTSIZE")) {
                            sscanf(line, "INPUTSIZE: %d\n", &layer->connected_layer->inputSize);
                        } else if(CMP_OPTION(line, "OUTPUTSIZE")) {
                            sscanf(line,"OUTPUTSIZE: %d\n", &layer->connected_layer->outputSize);
                        } else if(CMP_OPTION(line, "ACTIVATION")) {
                            layer->activation = PARSE_ACTIVATION(line);
                        } else if(CMP_OPTION(line, "RESHAPE")) {
                            sscanf(line,"RESHAPE: %d\n",&layer->connected_layer->need_reshape);
                        }
                        break;
                    case LAYER_TYPE_MAXPOOL:
                        if(CMP_OPTION(line, "SIZE")) {
                            sscanf(line,"SIZE: %d\n", &layer->maxpool_layer->size);
                        } else if(CMP_OPTION(line, "STRIDE")) {
                            sscanf(line,"STRIDE: %d %d\n", &layer->maxpool_layer->stride[0], &layer->maxpool_layer->stride[1]);
                        } else if(CMP_OPTION(line, "PAD")) {
                            sscanf(line,"PAD: %d %d %d %d\n", &layer->maxpool_layer->pad[0], &layer->maxpool_layer->pad[1], \
                                                            &layer->maxpool_layer->pad[2], &layer->maxpool_layer->pad[3]);
                        }
                        break;
                    case LAYER_TYPE_LRN_NORMALIZE:
                        if(CMP_OPTION(line, "SIZE")) {
                            sscanf(line,"SIZE: %d\n", &layer->lrn_layer->size);
                        } else if(CMP_OPTION(line, "ALPHA")) {
                            sscanf(line,"ALPHA: %f\n", &layer->lrn_layer->alpha);
                        } else if(CMP_OPTION(line, "BETA")) {
                            sscanf(line,"BETA: %f\n", &layer->lrn_layer->beta);
                        }
                        break;
                    case LAYER_TYPE_SOFTMAX:
                        break;
                    case LAYER_TYPE_UNKNOWN:
                        break;
                }
            }
        }
        fclose(layerfp);

        if(layer->type == LAYER_TYPE_CONV) {
            //determine output size
            if(layer->index == 0) {
                layer->output_w = (model->input_w + \
                                            layer->conv_layer->pad[0] + \
                                            layer->conv_layer->pad[1] - \
                                            layer->conv_layer->w) / \
					                        layer->conv_layer->stride[0] + 1;
                layer->output_h = (model->input_h + \
                                            layer->conv_layer->pad[2] + \
                                            layer->conv_layer->pad[3] - \
                                            layer->conv_layer->h) / \
                                            layer->conv_layer->stride[1] + 1;
                layer->output_c = layer->conv_layer->n;
            } else if(layer->index < 35){
                layer->output_w = (layers[layer->index - 1].output_w + \
                                            layer->conv_layer->pad[0] + \
                                            layer->conv_layer->pad[1] - \
                                            layer->conv_layer->w) / \
                                            layer->conv_layer->stride[0] + 1;
                layer->output_h = (layers[layer->index - 1].output_h + \
                                            layer->conv_layer->pad[2] + \
                                            layer->conv_layer->pad[3] - \
                                            layer->conv_layer->h) / \
                                            layer->conv_layer->stride[1] + 1;
                layer->output_c = layer->conv_layer->n;
            } else
            {
                //LOGI("bottom = %d",bottom[(layer->index - 35) / 2]);
                layer->output_w = (layers[bottom[(layer->index - 35) / 2]].output_w + \
                                            layer->conv_layer->pad[0] + \
                                            layer->conv_layer->pad[1] - \
                                            layer->conv_layer->w) / \
                                            layer->conv_layer->stride[0] + 1;
                //LOGI("W = %d",layer->output_w);
                layer->output_h = (layers[bottom[(layer->index - 35) / 2]].output_h + \
                                            layer->conv_layer->pad[2] + \
                                            layer->conv_layer->pad[3] - \
                                            layer->conv_layer->h) / \
                                            layer->conv_layer->stride[1] + 1;
                //LOGI("H = %d",layer->output_h);
                layer->output_c = layer->conv_layer->n;
                //LOGI("C = %d",layer->output_c);
            }


            //set forward function
            if(!layer->useINT8) {
              if (!layer->useGPU) {
                  if (layer->index == 0) {
                      layer->doFeedForward = doFeedForward_CONV_3_3_S2_NEON;
                  } else if (1 <= layer->index && layer->index <= 26) {
                      if (layer->index % 2 == 1) {
                          if (layer->conv_layer->stride[0] == 1) {
                              layer->doFeedForward = doFeedForward_CONV_DW_S1_NEON;
                          } else if (layer->conv_layer->stride[0] == 2) {
                              layer->doFeedForward = doFeedForward_CONV_DW_S2_NEON;
                          }
                      } else {
                          layer->doFeedForward = doFeedForward_CONV_1_1_NEON;
                      }
                  } else if (27 <= layer->index && layer->index <= 34) {
                      if (layer->index % 2 == 1) {
                          layer->doFeedForward = doFeedForward_CONV_1_1_NEON;
                      } else {
                          layer->doFeedForward = doFeedForward_CONV_3_3_S2_NEON;
                      }
                  } else {
                      layer->doFeedForward = doFeedForward_CONV_1_1_NEON;
                  }
              } else {   // layer->useGPU
                  if (layer->index == 0) {
                      layer->doFeedForward = doFeedForward_CONV_FIRST_GPU;
                  } else if (layer->index <= 26) {
                      if (layer->index % 2 == 1) {
                          layer->doFeedForward = doFeedForward_CONV_DW_GPU;
                          layer->dwFeedForward = doFeedForward_CONV_DW_MIX;
                      } else {
                          layer->doFeedForward = doFeedForward_CONV_1_1_GPU;
                          layer->dwFeedForward = doFeedForward_CONV_DW_1_1_MIX;
                      }
                  } else if (layer->index <= 34) {
                      if (layer->index % 2 == 1) {
                          layer->doFeedForward = doFeedForward_CONV_1_1_GPU;
                      } else {
                          layer->doFeedForward = doFeedForward_CONV_3_3_GPU;
                      }
                  } else {
                      layer->doFeedForward = doFeedForward_CONV_1_1_GPU;
                  }
              }
             }else{    // layer->useINT8
                if (!layer->useGPU) {
                    if (layer->index == 0) {
                        layer->doFeedForward = doFeedForward_CONV_3_3_S2_NEON_INT8;
                    } else if (1 <= layer->index && layer->index <= 26) {
                        if (layer->index % 2 == 1) {
                            if (layer->conv_layer->stride[0] == 1) {
                                layer->doFeedForward = doFeedForward_CONV_DW_S1_NEON_INT8;
                            } else if (layer->conv_layer->stride[0] == 2) {
                                layer->doFeedForward = doFeedForward_CONV_DW_S2_NEON_INT8;
                            }
                        } else {
                            layer->doFeedForward = doFeedForward_CONV_1_1_NEON_INT8;
                        }
                    } else if (27 <= layer->index && layer->index <= 34) {
                        if (layer->index % 2 == 1) {
                            layer->doFeedForward = doFeedForward_CONV_1_1_NEON_INT8;
                        } else {
                            layer->doFeedForward = doFeedForward_CONV_3_3_S2_NEON_INT8;
                        }
                    } else {
                        layer->doFeedForward = doFeedForward_CONV_1_1_NEON_INT8;
                    }
                } else {   // layer->useGPU  layer->useINT8
                    if (layer->index == 0) {
                        layer->doFeedForward = doFeedForward_CONV_FIRST_GPU_INT8;
                    } else if (layer->index <= 26) {
                        if (layer->index % 2 == 1) {
                            layer->doFeedForward = doFeedForward_CONV_DW_GPU_INT8;
                            layer->dwFeedForward = doFeedForward_CONV_DW_MIX;
                        } else {
                            layer->doFeedForward = doFeedForward_CONV_1_1_GPU_INT8;
                            layer->dwFeedForward = doFeedForward_CONV_DW_1_1_MIX;
                        }
                    } else if (layer->index <= 34) {
                        if (layer->index % 2 == 1) {
                            layer->doFeedForward = doFeedForward_CONV_1_1_GPU_INT8;
                        } else {
                            layer->doFeedForward = doFeedForward_CONV_3_3_GPU_INT8;
                        }
                    } else {
                        layer->doFeedForward = doFeedForward_CONV_1_1_GPU_INT8;
                    }
                }
            }

            //cpu - gpu -int8
//            if(layer->index <= 32&&layer->index >= 28){
//                if (layer->index % 2 == 1) {
//                    layer->doFeedForward = doFeedForward_CONV_1_1_GPU_INT8;
//                } else {
//                    layer->doFeedForward = doFeedForward_CONV_3_3_GPU_INT8;
//                }
//            }

            //读取weight scale
            if(layer->useINT8)   //INT8量化
            {
                char WeightScaleFilePath[256];
                strcpy(WeightScaleFilePath, fileNameBuf);
                strcat(WeightScaleFilePath, "_weight_scale");
                FILE *WeightScalefp = fopen(WeightScaleFilePath, "r");
                float * weight_scale_buffer=new float[layer->conv_layer->n];
                fread(weight_scale_buffer, sizeof(float), layer->conv_layer->n, WeightScalefp);
                layer->conv_layer->W_Scale = (float *)malloc(layer->conv_layer->n*sizeof(float));
                memcpy((void *)layer->conv_layer->W_Scale,(void *)weight_scale_buffer,layer->conv_layer->n*sizeof(float));

                delete[] weight_scale_buffer;
                fclose(WeightScalefp);
            }

            //LOAD BIAS & WEIGHTS DATA
            char biasFilePath[256];
            strcpy(biasFilePath, fileNameBuf);
            strcat(biasFilePath, "_bias");
            FILE *biasfp = fopen(biasFilePath, "r");
            float * bias_buffer=new float[layer->conv_layer->n];
            fread(bias_buffer, sizeof(float), layer->conv_layer->n, biasfp);

            if(layer->mixCPUGPU)
            {
                layer->conv_layer->bias = (float *)malloc(layer->conv_layer->n*sizeof(float));
                memcpy((void *)layer->conv_layer->bias,(void *)bias_buffer,layer->conv_layer->n*sizeof(float));
            }

            if(layer->useGPU)
            {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();
                //useGPU useINT8
                if(layer->useINT8)
                {
                    //有问题，bias全精度
                    layer->conv_layer->cl_bias = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->n * sizeof(signed char), //size in bytes
                            NULL,//buffer of data
                            &err);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    signed char *mappedBuffer = (signed char *) clEnqueueMapBuffer(openCLObjects->queue, \
                    layer->conv_layer->cl_bias, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
                    layer->conv_layer->n * sizeof(signed char), \
                    0, NULL, NULL, NULL);


                    clEnqueueUnmapMemObject(openCLObjects->queue, \
                    layer->conv_layer->cl_bias, \
                    mappedBuffer, \
                    0, NULL, NULL);
                }
                else//usGPU !useINT8
                {
                    layer->conv_layer->cl_bias = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->n * sizeof(float), //size in bytes
                            NULL,//buffer of data
                            &err);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    float *mappedBuffer = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    layer->conv_layer->cl_bias, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
                    layer->conv_layer->n * sizeof(float), \
                    0, NULL, NULL, NULL);

                    memcpy((void *) mappedBuffer, (void *) bias_buffer,
                           layer->conv_layer->n * sizeof(cl_float));

                    clEnqueueUnmapMemObject(openCLObjects->queue, \
                    layer->conv_layer->cl_bias, \
                    mappedBuffer, \
                    0, NULL, NULL);

                    if (layer->useHalf)//useGPU useHalf
                    {
                        cl_mem cl_bias_half = clCreateBuffer(
                                openCLObjects->context,
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                layer->conv_layer->n * sizeof(cl_half), //size in bytes
                                NULL,//buffer of data
                                &err);

                        err = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0,
                                             sizeof(cl_mem), &layer->conv_layer->cl_bias);
                        err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1,
                                              sizeof(cl_mem), &cl_bias_half);
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                        size_t convertSize[1] = {(size_t) layer->conv_layer->n};
                        err = clEnqueueNDRangeKernel(
                                openCLObjects->queue,
                                openCLObjects->convert_float_to_half_kernel.kernel,
                                1,
                                0,
                                convertSize,
                                0,
                                0, 0, 0
                        );
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
                        err = clFinish(openCLObjects->queue);
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                        clReleaseMemObject(layer->conv_layer->cl_bias);

                        layer->conv_layer->cl_bias = cl_bias_half;

                        LOGI("bias float to half\n");
                    }
                }
            }
            delete[] bias_buffer;
            fclose(biasfp);

            char wFilePath[256];
            strcpy(wFilePath, fileNameBuf);
            strcat(wFilePath, "_weight");
            FILE *wfp = fopen(wFilePath, "r");
            float * weight_buffer=new float[layer->conv_layer->n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w];
            fread(weight_buffer, sizeof(float), layer->conv_layer->n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w, wfp);

            if(layer->mixCPUGPU)
            {

                if(layer->useINT8)   //INT8量化
                {
                    //量化模型weight参数需要量化表，加入比例后再量化
                    layer->conv_layer->W_S8 =(signed char *) malloc(layer->conv_layer->n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w * sizeof(signed char));
                    //可利用Neon优化
                    //每个卷积核一个量化比例
                    for (int n = 0; n < layer->conv_layer->n; ++n) {
                        for (int j = 0; j < layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w; ++j) {
                            layer->conv_layer->W_S8[j + n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w] =
                                    float2int8(weight_buffer[j + n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w]*layer->conv_layer->W_Scale[n]);
                        }
                    }
                }
                else
                {
                    //非对齐的weight
                    layer->conv_layer->W = (float *)malloc(layer->conv_layer->n*layer->conv_layer->w*layer->conv_layer->h*layer->conv_layer->c * sizeof(float));
                    memcpy((void*)(layer->conv_layer->W),(void*)(weight_buffer),layer->conv_layer->n*layer->conv_layer->c*layer->conv_layer->h*layer->conv_layer->w* sizeof(float));
                }
            }

            if(layer->useGPU) {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                if(layer->useINT8){   //useGPU useINT8
                    layer->conv_layer->cl_W = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c *
                            layer->conv_layer->n * sizeof(signed char), //size in bytes
                            NULL,//buffer of data
                            &err);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    signed char *mappedBuffer = (signed char *) clEnqueueMapBuffer(openCLObjects->queue, \
                    layer->conv_layer->cl_W, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n * sizeof(signed char), \
                    0, NULL, NULL, NULL);

                    for (int k = 0; k < layer->conv_layer->n; k++) {
                        for (int c = 0; c < layer->conv_layer->c; c++) {
                            for (int h = 0; h < layer->conv_layer->h; h++) {
                                for (int w = 0; w < layer->conv_layer->w; w++) {
                                    int buf_idx = getIndexFrom4D(layer->conv_layer->n,
                                                                 layer->conv_layer->c,
                                                                 layer->conv_layer->h,
                                                                 layer->conv_layer->w, k, c, h, w);
                                    int new_idx = getIndexFrom4D(layer->conv_layer->n,
                                                                 layer->conv_layer->h,
                                                                 layer->conv_layer->w,
                                                                 layer->conv_layer->c, k, h, w, c);
                                    mappedBuffer[new_idx] = float2int8(weight_buffer[buf_idx] * layer->conv_layer->W_Scale[k]);
                                }
                            }
                        }
                    }

                    clEnqueueUnmapMemObject(openCLObjects->queue, \
                    layer->conv_layer->cl_W, \
                    mappedBuffer, \
                    0, NULL, NULL);


                }else {    //useGPU !useINT8
                    layer->conv_layer->cl_W = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c *
                            layer->conv_layer->n * sizeof(float), //size in bytes
                            NULL,//buffer of data
                            &err);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    float *mappedBuffer = (float *) clEnqueueMapBuffer(openCLObjects->queue, \
                    layer->conv_layer->cl_W, \
                    CL_TRUE, CL_MAP_WRITE, \
                    0, \
     layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n *
     sizeof(float), \
                    0, NULL, NULL, NULL);

                    for (int k = 0; k < layer->conv_layer->n; k++) {
                        for (int c = 0; c < layer->conv_layer->c; c++) {
                            for (int h = 0; h < layer->conv_layer->h; h++) {
                                for (int w = 0; w < layer->conv_layer->w; w++) {
                                    int buf_idx = getIndexFrom4D(layer->conv_layer->n,
                                                                 layer->conv_layer->c,
                                                                 layer->conv_layer->h,
                                                                 layer->conv_layer->w, k, c, h, w);
                                    int new_idx = getIndexFrom4D(layer->conv_layer->n,
                                                                 layer->conv_layer->h,
                                                                 layer->conv_layer->w,
                                                                 layer->conv_layer->c, k, h, w, c);
                                    mappedBuffer[new_idx] = weight_buffer[buf_idx];
                                }
                            }
                        }
                    }

                    clEnqueueUnmapMemObject(openCLObjects->queue, \
                    layer->conv_layer->cl_W, \
                    mappedBuffer, \
                    0, NULL, NULL);

                    if (layer->useHalf == 1) {   //useGPU useHalf
                        cl_mem cl_W_half = clCreateBuffer(
                                openCLObjects->context,
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c *
                                layer->conv_layer->n * sizeof(cl_half), //size in bytes
                                NULL,//buffer of data
                                &err);

                        err = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0,
                                             sizeof(cl_mem), &layer->conv_layer->cl_W);
                        err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1,
                                              sizeof(cl_mem), &cl_W_half);
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                        size_t convertSize[1] = {
                                (size_t) layer->conv_layer->w * layer->conv_layer->h *
                                layer->conv_layer->c * layer->conv_layer->n};
                        err = clEnqueueNDRangeKernel(
                                openCLObjects->queue,
                                openCLObjects->convert_float_to_half_kernel.kernel,
                                1,
                                0,
                                convertSize,
                                0,
                                0, 0, 0
                        );
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                        err = clFinish(openCLObjects->queue);
                        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                        clReleaseMemObject(layer->conv_layer->cl_W);

                        layer->conv_layer->cl_W = cl_W_half;
                    }
                }
            }
            delete[] weight_buffer;
            fclose(wfp);
        }


        if(layer->type == LAYER_TYPE_FULLY_CONNECTED) {
            layer->output_w = 1;
            layer->output_h = 1;
            layer->output_c = layer->connected_layer->outputSize;

            layer->connected_layer->weightSize = layer->connected_layer->inputSize * layer->connected_layer->outputSize;

            //LOAD BIAS AND WEIGHTS DATA
            char biasFilePath[256];
            strcpy(biasFilePath, fileNameBuf);
            strcat(biasFilePath, "_bias");
            FILE *biasfp = fopen(biasFilePath, "r");
            if(!layer->useGPU) {
                layer->connected_layer->bias = (float *)calloc(layer->connected_layer->outputSize, sizeof(float));
                fread(layer->connected_layer->bias, sizeof(float), layer->connected_layer->outputSize, biasfp);

            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->connected_layer->cl_bias = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->connected_layer->outputSize * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->connected_layer->cl_bias, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->connected_layer->outputSize * sizeof(float), \
					0, NULL, NULL, NULL);

                fread(mappedBuffer, sizeof(float), layer->connected_layer->outputSize, biasfp);

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->connected_layer->cl_bias, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf) {
                    cl_mem cl_bias_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->connected_layer->outputSize * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->connected_layer->cl_bias);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_bias_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t) layer->connected_layer->outputSize};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->connected_layer->cl_bias);

                    layer->connected_layer->cl_bias = cl_bias_half;
                }
            }
            fclose(biasfp);

            char wFilePath[256];
            strcpy(wFilePath, fileNameBuf);
            strcat(wFilePath, "_weight");
            FILE *wfp = fopen(wFilePath, "r");
            if(!layer->useGPU) {
                layer->connected_layer->W = (float *) calloc(\
                    layer->connected_layer->weightSize, sizeof(float));
                fread(layer->connected_layer->W, sizeof(float),
                      layer->connected_layer->weightSize,
                      wfp);
            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->connected_layer->cl_W = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->connected_layer->weightSize * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->connected_layer->cl_W, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->connected_layer->weightSize * sizeof(float), \
					0, NULL, NULL, NULL);

                if(!layer->connected_layer->need_reshape) {
                    //file is formatted [(c x h x w) x outputsize]
                    //this is for LRCN
                    float * buffer = (float *)malloc(layer->connected_layer->outputSize * sizeof(float));
                    int input_h = (layer->index == 0) ? model->input_h : layers[layer->index - 1].output_h;
                    int input_w = (layer->index == 0) ? model->input_w : layers[layer->index - 1].output_w;
                    int input_c = (layer->index == 0) ? model->input_c : layers[layer->index - 1].output_c;

                    for(int c = 0 ; c < input_c ; c++) {
                        for(int h = 0 ; h < input_h ; h++) {
                            for(int w = 0 ; w < input_w ; w++) {
                                fread(buffer, sizeof(float), layer->connected_layer->outputSize, wfp);
                                for(int n = 0 ; n < layer->connected_layer->outputSize ; n++) {
                                    int idx = getIndexFrom4D(layer->connected_layer->outputSize, input_h, input_w, input_c, n, h, w, c);
                                    mappedBuffer[idx] = buffer[n];
                                }
                            }
                        }
                    }
                    free(buffer);
                } else {
                    //file is formatted [outputsize x (c x h x w)]
                    int input_h = (layer->index == 0) ? model->input_h : layers[layer->index - 1].output_h;
                    int input_w = (layer->index == 0) ? model->input_w : layers[layer->index - 1].output_w;
                    int input_c = (layer->index == 0) ? model->input_c : layers[layer->index - 1].output_c;

                    int size = input_h * input_w * input_c;
                    float *buffer = (float *)malloc(size * sizeof(float));
                    for(int n = 0 ; n < layer->connected_layer->outputSize ; n++) {
                        fread(buffer, sizeof(float), size, wfp); //[c x h x w]
                        //need to convert to h x w x c
                        int f_idx = 0;
                        for(int c = 0 ; c < input_c ; c++) {
                            for(int h = 0 ; h < input_h ; h++) {
                                for(int w = 0 ; w < input_w ; w++) {
                                    int idx = getIndexFrom4D(layer->connected_layer->outputSize, input_h, input_w, input_c, n, h, w, c);
                                    mappedBuffer[idx] = buffer[f_idx];
                                    f_idx++;
                                }
                            }
                        }
                    }
                    free(buffer);
                }

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->connected_layer->cl_W, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf) {
                    cl_mem cl_W_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->connected_layer->weightSize * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->connected_layer->cl_W);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_W_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t) layer->connected_layer->weightSize};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->connected_layer->cl_W);

                    layer->connected_layer->cl_W = cl_W_half;
                }
            }
            fclose(wfp);
        }

        if(layer->type == LAYER_TYPE_MAXPOOL) {
            layer->output_w = 1 + (layers[layer->index - 1].output_w + layer->maxpool_layer->pad[0] + layer->maxpool_layer->pad[1] - layer->maxpool_layer->size) / layer->maxpool_layer->stride[0];
            layer->output_h = 1 + (layers[layer->index - 1].output_h + layer->maxpool_layer->pad[2] + layer->maxpool_layer->pad[3] - layer->maxpool_layer->size) / layer->maxpool_layer->stride[1];
            layer->output_c = layers[layer->index - 1].output_c;
        }

        if(layer->type == LAYER_TYPE_SOFTMAX) {
            layer->output_w = 1;
            layer->output_h = 1;
            layer->output_c = layers[layer->index - 1].output_c;
        }

        if(layer->type == LAYER_TYPE_LRN_NORMALIZE) {
            layer->output_w = layers[layer->index - 1].output_w;
            layer->output_h = layers[layer->index - 1].output_h;
            layer->output_c = layers[layer->index - 1].output_c;
        }

        int input_w = (i == 1) ? model->input_w : layers[i - 2].output_w;
        int input_h = (i == 1) ? model->input_h : layers[i - 2].output_h;
        int input_c = (i == 1) ? model->input_c : layers[i - 2].output_c;

        LOGI("Layer %d has input[%d %d %d] and output [%d %d %d]",(i), \
             input_c, input_h, input_w, layer->output_c, layer->output_h, layer->output_w);

    }

    model->layers[model->nLayers].detection_output_layer=(cnn_layer_detection_output *) calloc(1, sizeof(cnn_layer_detection_output));

    char modelDir[256];
    sprintf(modelDir, "%s", modelDirPath);
    if(strcmp(modelDir,"sdcard/MobileNetSSD-DeepSense-android") == 0)
    {
        //object
        model->layers[model->nLayers].detection_output_layer->confidence_threshold = 0.25;
        model->layers[model->nLayers].detection_output_layer->keep_top_k =30;
        model->layers[model->nLayers].detection_output_layer->nms_top_k = 100;
        model->layers[model->nLayers].detection_output_layer->nms_threshold = 0.449999988079;
        model->layers[model->nLayers].detection_output_layer->num_class = 21;
        if(model->useGPU)
        {
            model->layers[model->nLayers].detectOutputForward = doFeedForward_DETECTION_OUTPUT_GPU;
        }
        else
        {
            model->layers[model->nLayers].detectOutputForward = doFeedForward_DETECTION_OUTPUT;
        }
    }
    else
    {
        //face
        model->layers[model->nLayers].detection_output_layer->confidence_threshold = 0.25;
        model->layers[model->nLayers].detection_output_layer->keep_top_k =100;
        model->layers[model->nLayers].detection_output_layer->nms_top_k = 100;
        model->layers[model->nLayers].detection_output_layer->nms_threshold = 0.449999988079;
        model->layers[model->nLayers].detection_output_layer->num_class = 2;
        if(model->useGPU)
        {
            model->layers[model->nLayers].detectOutputForward = doFeedForward_DETECTION_OUTPUT_FACE_GPU;
        }
        else
        {
            model->layers[model->nLayers].detectOutputForward = doFeedForward_DETECTION_OUTPUT_FACE;
        }
    }

    return model;
}

void cnn_free(cnn *model) {
    int i;
    for(i = 0 ; i < model->nLayers ; i++)
    {
        cnn_layer *layer = &model->layers[i];
        if(layer->type == LAYER_TYPE_CONV)
        {
            if( layer->mixCPUGPU )
            {
                if(layer->useINT8)
                {
                    free(layer->conv_layer->W_Scale);
                    free(layer->conv_layer->W_S8);
                }
                else
                {
                    free(layer->conv_layer->W);
                }
                free(layer->conv_layer->bias);
            }
            if(layer->useGPU)
            {
                if(layer->useINT8)
                {
                    free(layer->conv_layer->W_Scale);
                }
                clReleaseMemObject(layer->conv_layer->cl_W);
                clReleaseMemObject(layer->conv_layer->cl_bias);
            }
            free(layer->conv_layer);
        }
        else if(layer->type == LAYER_TYPE_FULLY_CONNECTED)
        {
            if(!model->useGPU) {
                free(layer->connected_layer->bias);
                free(layer->connected_layer->W);
            } else {
                clReleaseMemObject(layer->conv_layer->cl_W);
                clReleaseMemObject(layer->conv_layer->cl_bias);
            }
            free(layer->connected_layer);
        } else if(layer->type == LAYER_TYPE_MAXPOOL)
        {
            free(layer->maxpool_layer);
        } else if(layer->type == LAYER_TYPE_LRN_NORMALIZE)
        {
            free(layer->lrn_layer);
        }
    }

    if(model->averageImage != NULL)
        free(model->averageImage);

    free(model->layers);
    free(model);
}