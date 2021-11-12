#include <jni.h>
#include <android/log.h>
#include <string>
#include <deepsense_lib.hpp>
#include <deepsense_internal_lib.hpp>
#include <predefine.hpp>
#include <utilities.hpp>
#include <basic_functions.hpp>
#include <classifier.hpp>
#include <clio.hpp>
#ifndef LOG_PRINT
#define DEBUG_TAG "NDK_SampleActivity"
#define LOG_TAG "hellojni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

cnn *model = NULL;
OpenCLObjects openCLObjects;

OpenCLObjects *getOpenClObject() {
    return &openCLObjects;
}

cnn *getModel() {
    return model;
}

extern "C" void Java_com_lanytek_deepsensev3_NativeMethod_InitGPU(
        JNIEnv* env,
        jobject thiz,
        jstring model_dir_path,
        jstring packageName
) {

    LOGI("1 can you can a can like can can a can \n");

    //init GPU first
    const char *packageNameStr = env->GetStringUTFChars(packageName, 0);
    init_OpenCL(CL_DEVICE_TYPE_GPU, openCLObjects, packageNameStr);
    env->ReleaseStringUTFChars(packageName, packageNameStr);

    LOGI("2 can you can a can like can can a can \n");

    //init model
    const char *modelPath = env->GetStringUTFChars(model_dir_path, 0);
    if(model != NULL) {
        cnn_free(model);
    }

    model = cnn_loadModel(modelPath, 0);

    LOGI("3 can you can a can like can can a can \n");

    env->ReleaseStringUTFChars(model_dir_path, modelPath);
}

extern "C" jfloatArray Java_com_lanytek_deepsensev3_NativeMethod_GetInferrence(
        JNIEnv* env,
        jobject thisObject,
        jfloatArray input
) {
    if(model == NULL)
        return NULL;

    LOGI("MobileNetSSD input w=%d h=%d c=%d",model->input_w, model->input_h, model->input_c);
    cnn_frame *frame;
    jfloat * data = env->GetFloatArrayElements(input, 0);

    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;


    //初始化数据层
    //cpu-float
    if(getModel()->useINT8==0 && getModel()->useGPU==0)
    {
        frame = frame_init(model->input_w, model->input_h, model->input_c, 0);
        for (int i = 0; i < frame->c; ++i)
        {
            // memcpy((void *) (frame->data + i * alignSize(frame->w * frame->h * sizeof(float), 16) / sizeof(float)), (void *) (data + i * frame->w * frame->h), frame->w * frame->h * sizeof(float));
            for(int j = 0;j<frame->w * frame->h;j++)
            {
                (frame->data + (i) * alignSize(frame->w * frame->h * sizeof(float), 16) / sizeof(float))[j] = *(data + j * frame->c + i);
            }
        }
    }
    else if(getModel()->useGPU==0 && getModel()->useINT8==1)
    {
        frame = frame_init(model->input_w,model->input_h,model->input_c,1);
        for (int i = 0; i < frame->c; ++i) {
            for (int j = 0; j < frame->w * frame->h; ++j) {
                frame->data_s8[j + i * alignSize(frame->w * frame->h * sizeof(signed char), 16) / sizeof(signed char)] =  float2int8(data[j*3 +i] * model->layers[0].Scale_in);
            }
        }
    }
    else if(getModel()->useGPU==1 && getModel()->useINT8==0)
    {
        frame = frame_init_gpu(model->input_w, model->input_h, model->input_c);
        float *cl_ptr = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, \
					0, \
					model->input_w * model->input_h * model->input_c * sizeof(float), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
        memcpy(cl_ptr, data, model->input_w * model->input_h * model->input_c * sizeof(float));

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					cl_ptr, \
					0, NULL, NULL);

        frame = frame_convert_to_gpu_half(frame);
    }
    else if(getModel()->useGPU==1 && getModel()->useINT8==1)
    {
        //blob量化时没有乘blob_scale
        frame = frame_init_gpu_int8(model->input_w, model->input_h, model->input_c);
        signed char *cl_ptr = ( signed char *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					model->input_w * model->input_h * model->input_c * sizeof(signed char), \
					0, NULL, NULL, &err);

        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
        for (int j = 0; j < frame->w * frame->h*frame->c; ++j)
        {
            cl_ptr[j] = data[j];
        }
        clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					cl_ptr, \
					0, NULL, NULL);//解除映射
    }


    env->ReleaseFloatArrayElements(input, data, 0);

    frame = cnn_doClassification(frame, model);

    if(frame != NULL) {
        int outputSize = frame->w*frame->h*frame->c;
        jfloatArray resultArr = env->NewFloatArray(outputSize);
        env->SetFloatArrayRegion(resultArr, 0, outputSize, frame->data);
        frame_free_not_align(frame);

        LOGI("return success");
        return  resultArr;
    }
    else
    {
        jfloatArray resultArr = env->NewFloatArray(1);
        return  resultArr;
    }
}

extern "C" void Java_com_lanytek_deepsensev3_NativeMethod_ReleaseCNN(
        JNIEnv* env,
        jobject thisObject
){
    shutdown_OpenCL(*getOpenClObject());
    cnn_free(getModel());
}



