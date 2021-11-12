#include <classifier.hpp>
#include <basic_functions.hpp>
#include <clio.hpp>
#include <malloc.h>
#include <stdio.h>
#include <deepsense_internal_lib.hpp>
#include <deepsense_lib.hpp>
#include <layers/conv_layer.hpp>
#include <layers/detection_output.hpp>
#include <jni.h>
//LOGI
#include <android/log.h>
//std
#include <vector>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>

//openmp
#include <omp.h>


static int get_max_freq_khz(int cpuid)
{
    // first try, for all possible cpu
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
    {
        // second try, for online cpu
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
        fp = fopen(path, "rb");

        if (!fp)
        {
            // third try, for online cpu
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
            fp = fopen(path, "rb");

            if (!fp)
                return -1;

            int max_freq_khz = -1;
            fscanf(fp, "%d", &max_freq_khz);

            fclose(fp);

            return max_freq_khz;
        }
    }

    int max_freq_khz = 0;
    while (!feof(fp))
    {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1)
            break;

        if (freq_khz > max_freq_khz)
            max_freq_khz = freq_khz;
    }

    fclose(fp);

    return max_freq_khz;
}

static int set_sched_affinity(const std::vector<int>& cpuids)
{
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
    typedef struct
    {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
    pid_t pid = gettid();
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i=0; i<(int)cpuids.size(); i++)
    {
        CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret)
    {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

static int sort_cpuid_by_max_frequency(std::vector<int>& cpuids, int* little_cluster_offset)
{
    const int cpu_count = cpuids.size();

    *little_cluster_offset = 0;

    if (cpu_count == 0)
        return 0;

    std::vector<int> cpu_max_freq_khz;
    cpu_max_freq_khz.resize(cpu_count);

    for (int i=0; i<cpu_count; i++)
    {
        int max_freq_khz = get_max_freq_khz(i);

        LOGI("%d max freq = %d khz\n", i, max_freq_khz);

        cpuids[i] = i;
        cpu_max_freq_khz[i] = max_freq_khz;
    }

    // sort cpuid as big core first
    // simple bubble sort
    for (int i=0; i<cpu_count; i++)
    {
        for (int j=i+1; j<cpu_count; j++)
        {
            if (cpu_max_freq_khz[i] < cpu_max_freq_khz[j])
            {
                // swap
                int tmp = cpuids[i];
                cpuids[i] = cpuids[j];
                cpuids[j] = tmp;

                tmp = cpu_max_freq_khz[i];
                cpu_max_freq_khz[i] = cpu_max_freq_khz[j];
                cpu_max_freq_khz[j] = tmp;
            }
        }
    }

    // SMP
    int mid_max_freq_khz = (cpu_max_freq_khz.front() + cpu_max_freq_khz.back()) / 2;
    if (mid_max_freq_khz == cpu_max_freq_khz.back())
        return 0;

    for (int i=0; i<cpu_count; i++)
    {
        if (cpu_max_freq_khz[i] < mid_max_freq_khz)
        {
            *little_cluster_offset = i;
            break;
        }
    }

    return 0;
}


//主进程与子进程TCP连接
int server_fd, client_fd;
cnn_frame * cnn_doClassification(cnn_frame *frame, cnn *model) {

    cnn_frame *result = frame;

    //根据CPU各个core的频率将core的ID序列降序排列
    int num_core_used = 4;
    int num_core_total = 8;
    static std::vector<int> sorted_cpuids(num_core_total);
    static int little_cluster_offset = 0;
    sort_cpuid_by_max_frequency(sorted_cpuids, &little_cluster_offset);
    std::vector<int> cpuids;
    if(num_core_used == 1)
    {
        cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() + 1);
    }
    else if(num_core_used == 2)
    {
        cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() + little_cluster_offset);
    }
    else if(num_core_used == 4)
    {
        cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() + little_cluster_offset);
    }
    else if(num_core_used == 6)
    {
        cpuids = sorted_cpuids;
    }
    else if(num_core_used == 8)
    {
        cpuids = sorted_cpuids;
    }
    for (int n = 0; n < num_core_used; ++n)
    {
        LOGI("descent %d ",cpuids[n]);
    }
    LOGI("little_cluster_offset %d ",little_cluster_offset);
    //选择OpenMP的进程数，并将core的ID绑定到进程上
    int num_threads = cpuids.size();
    omp_set_num_threads(num_core_used);
    std::vector<int> ssarets(num_core_used, 0);
    #pragma omp parallel for
    for (int j=0; j<num_core_used; j++)
    {
        ssarets[j] = set_sched_affinity(cpuids);
    }
    for (int j=0; j<num_core_used; j++)
    {
        LOGI("ssarets %d ",ssarets[j]);
    }

    int total_count = 1;
    double totalTime = 0;
    double t0,t1;
    double global_t0 = get_timestamp();

    //子进程利用Neon指令计算
/*
    //*********************创建共享内存*************************
    int shm_id1,shm_id2;
    unsigned char *shm_buf1;   //指向共享内存 存储输入数据
    unsigned char *shm_buf2;   //指向共享内存 存储输出参数
    unsigned char **shm_data1; //指向共享内存中字节对齐的地址
    unsigned char **shm_data2; //指向共享内存中字节对齐的地址
    float *shm_buffer1;
    float *shm_buffer2;

    //计算字节对齐的共享内存空间需要的大小
    int MaxFrame_w=150;
    int MaxFrame_h=150;
    int MaxFrame_c=64;
    int elementsize= sizeof(float);
    int *refcount1;
    int *refcount2;

    int cstep = alignSize(MaxFrame_w*MaxFrame_h*elementsize,16)/elementsize;
    int total=cstep*MaxFrame_c;
    int totalsize = alignSize(total*elementsize,4);
    int allsize=totalsize+(int) sizeof(*refcount1)+ sizeof(void *)+MALLOC_ALIGN;

    int shm_ret = create_shared_memory("SHM_1", allsize, shm_buf1, shm_id1);
    if (shm_ret < 0) {
        LOGI("CREATE SHARED MEMORY ERROR!!!\n");
    }
    shm_ret = create_shared_memory("SHM_2", allsize, shm_buf2, shm_id2);
    if (shm_ret < 0) {
        LOGI("CREATE SHARED MEMORY ERROR!!!\n");
    }
    //*********************创建共享内存*************************

    //*********************派生子进程*************************
    if(fork()==0){
        LOGI("ChildProcessCreate!!!\n");
        char socketbuffer[2];
        int LayerNum=0;
        SocketClientCreate();

        //子进程中，获取共享内存的对齐指针
        shm_data1= alignPtr((unsigned char **) shm_buf1 + 1, MALLOC_ALIGN);
        shm_data1[-1]=shm_buf1;
        refcount1 = (int *) (((unsigned char *) shm_data1) + totalsize);
        *refcount1 = 1;

        shm_data2= alignPtr((unsigned char **) shm_buf2 + 1, MALLOC_ALIGN);
        shm_data2[-1]=shm_buf2;
        refcount2 = (int *) (((unsigned char *) shm_data2) + totalsize);
        *refcount2 = 1;
        //测试完成，shm_buf1指向共享内存，不需要open共享内存了
        shm_buffer1=(float *)shm_data1;
        shm_buffer2=(float *)shm_data2;
        //共享内存指针获取到了

        while(1) {
            //socket获取同步信息，和需要执行的层数
            SocketRecvALL( socketbuffer, 1);

            LayerNum = socketbuffer[0] - 'A'+1;  //子进程执行第三层，则LayerNum=3
            cnn_layer *layer1 = &model->layers[LayerNum-1];
            cnn_layer *lastlayer=&model->layers[LayerNum-2];
            cnn_frame *frame1 = frame_init_share(lastlayer->output_w, lastlayer->output_h, lastlayer->output_c);
            frame1->data=shm_buffer1;
            int loop = 1;
            t0 = get_timestamp();
            for (int i = 0; i < loop; ++i) {
                doFeedForward_CONV_ASM_NEON_MIX(frame1, layer1,shm_buffer2);
            }
            SocketSendALL(socketbuffer, 1);  //计算完成，发送同步信息给主进程
            t1 = get_timestamp();
            double milsecs = (t1 - t0) / 1000.0L / loop;
            LOGI("ChildProcessTime:%f ms!!!\n", milsecs);
            if(LayerNum==27)
            {
                LOGI("ChildProcessEND!!!\n");
                break;
            }
        }
        SocketClientClose();
        return 0;
    }
    //*********************派生子进程*************************

    SocketServerCreate();

    //***************父进程获取共享内存的对齐指针****************
    shm_data1= alignPtr((unsigned char **) shm_buf1 + 1, MALLOC_ALIGN);
    shm_data1[-1]=shm_buf1;
    refcount1 = (int *) (((unsigned char *) shm_data1) + totalsize);
    *refcount1 = 1;

    shm_data2= alignPtr((unsigned char **) shm_buf2 + 1, MALLOC_ALIGN);
    shm_data2[-1]=shm_buf2;
    refcount2 = (int *) (((unsigned char *) shm_data2) + totalsize);
    *refcount2 = 1;
    shm_buffer1=(float *)shm_data1;
    shm_buffer2=(float *)shm_data2;
    //***************父进程获取共享内存的对齐指针****************
*/

    OpenCLObjects *openCLObjects = getOpenClObject();

    cnn_frame *loc_result[6];
    cnn_frame *conf_result[6];

    for (int m = 0; m < total_count; ++m) {

        cl_int err;
        cl_event event;

        //23、27、29、31、33、35层的卷积计算结果
        cnn_frame * loc_franme[6];
        cnn_frame * conf_franme[6];

        LOGI("model->nLayers = %d",model->nLayers);

        for (int i = 0; i < 35; i++)
        {
            cnn_layer *layer = &model->layers[i];

            t0 = get_timestamp();
            layer->num_threads = num_threads;
            /*
            if(i>=1 && i<=26 && i%2==1)
                result = layer->dwFeedForward(result, layer,shm_buffer1,client_fd,i+1);         //dw卷积
            else if(i>=1 && i<=26 && i%2==0)
                result = layer->dwFeedForward(result, layer,shm_buffer2,client_fd,i+1);         //点卷积
            else
                result = layer->doFeedForward(result, layer);
                */

            //重量化
            if(i!=0 && layer->useINT8==1 && layer->useGPU==0)
            {
                result->data_s8 = requantize_arm((int*)result->data_s8,result->c,result->h,result->w,model->layers[i -1].conv_layer->bias,
                                                 model->layers[i-1].Scale_in,model->layers[i].Scale_in,model->layers[i-1].conv_layer->W_Scale);
                //激活函数要求INT8的帧 先重量化再激活
                activate_neon_RELU_INT8(result);
            }

            result = layer->doFeedForward(result, layer);

            t1 = get_timestamp();
            double milsecs = (t1 - t0) / 1000.0L;
            LOGI("Processed layer %d in %f ms\n", (i + 1), milsecs);

            //保存中间结果
            switch (i) {
                case 22:
                    //这里的clone函数有修改 INT32
                    loc_franme[0] = frame_clone(result);
                    conf_franme[0] = frame_clone(result);
                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[0]->data_s8 = requantize_arm((int*)loc_franme[0]->data_s8,loc_franme[0]->c,loc_franme[0]->h,loc_franme[0]->w,
                                                                model->layers[22].conv_layer->bias,model->layers[22].Scale_in,model->layers[35].Scale_in,model->layers[22].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[0]);

                        conf_franme[0]->data_s8 = requantize_arm((int*)conf_franme[0]->data_s8,conf_franme[0]->c,conf_franme[0]->h,conf_franme[0]->w,
                                                                 model->layers[22].conv_layer->bias,model->layers[22].Scale_in,model->layers[36].Scale_in,model->layers[22].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[0]);
                    }
                    break;
                case 26:
                    loc_franme[1] = frame_clone(result);
                    conf_franme[1] = frame_clone(result);

                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[1]->data_s8 = requantize_arm((int*)loc_franme[1]->data_s8,loc_franme[1]->c,loc_franme[1]->h,loc_franme[1]->w,
                                                                model->layers[26].conv_layer->bias,model->layers[26].Scale_in,model->layers[37].Scale_in,model->layers[26].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[1]);

                        conf_franme[1]->data_s8 = requantize_arm((int*)conf_franme[1]->data_s8,conf_franme[1]->c,conf_franme[1]->h,conf_franme[1]->w,
                                                                 model->layers[26].conv_layer->bias,model->layers[26].Scale_in,model->layers[38].Scale_in,model->layers[26].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[1]);
                    }
                    break;
                case 28:
                    loc_franme[2] = frame_clone(result);
                    conf_franme[2] = frame_clone(result);

                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[2]->data_s8 = requantize_arm((int*)loc_franme[2]->data_s8,loc_franme[2]->c,loc_franme[2]->h,loc_franme[2]->w,
                                                                model->layers[28].conv_layer->bias,model->layers[28].Scale_in,model->layers[39].Scale_in,model->layers[28].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[2]);

                        conf_franme[2]->data_s8 = requantize_arm((int*)conf_franme[2]->data_s8,conf_franme[2]->c,conf_franme[2]->h,conf_franme[2]->w,
                                                                 model->layers[28].conv_layer->bias,model->layers[28].Scale_in,model->layers[40].Scale_in,model->layers[28].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[2]);
                    }
                    break;
                case 30:
                    loc_franme[3] = frame_clone(result);
                    conf_franme[3] = frame_clone(result);

                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[3]->data_s8 = requantize_arm((int*)loc_franme[3]->data_s8,loc_franme[3]->c,loc_franme[3]->h,loc_franme[3]->w,
                                                                model->layers[30].conv_layer->bias,model->layers[30].Scale_in,model->layers[41].Scale_in,model->layers[30].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[3]);

                        conf_franme[3]->data_s8 = requantize_arm((int*)conf_franme[3]->data_s8,conf_franme[3]->c,conf_franme[3]->h,conf_franme[3]->w,
                                                                 model->layers[30].conv_layer->bias,model->layers[30].Scale_in,model->layers[42].Scale_in,model->layers[30].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[3]);
                    }
                    break;
                case 32:
                    loc_franme[4] = frame_clone(result);
                    conf_franme[4] = frame_clone(result);

                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[4]->data_s8 = requantize_arm((int*)loc_franme[4]->data_s8,loc_franme[4]->c,loc_franme[4]->h,loc_franme[4]->w,
                                                                model->layers[32].conv_layer->bias,model->layers[32].Scale_in,model->layers[43].Scale_in,model->layers[32].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[4]);

                        conf_franme[4]->data_s8 = requantize_arm((int*)conf_franme[4]->data_s8,conf_franme[4]->c,conf_franme[4]->h,conf_franme[4]->w,
                                                                 model->layers[32].conv_layer->bias,model->layers[32].Scale_in,model->layers[44].Scale_in,model->layers[32].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[4]);
                    }
                    break;
                case 34:
                    loc_franme[5] = frame_clone(result);
                    conf_franme[5] = frame_clone(result);

                    //在这里重量化 和RELU
                    if(layer->useINT8)
                    {
                        loc_franme[5]->data_s8 = requantize_arm((int*)loc_franme[5]->data_s8,loc_franme[5]->c,loc_franme[5]->h,loc_franme[5]->w,
                                                                model->layers[34].conv_layer->bias,model->layers[34].Scale_in,model->layers[45].Scale_in,model->layers[34].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(loc_franme[5]);

                        conf_franme[5]->data_s8 = requantize_arm((int*)conf_franme[5]->data_s8,conf_franme[5]->c,conf_franme[5]->h,conf_franme[5]->w,
                                                                 model->layers[34].conv_layer->bias,model->layers[34].Scale_in,model->layers[45].Scale_in,model->layers[34].conv_layer->W_Scale);
                        activate_neon_RELU_INT8(conf_franme[5]);
                    }
                    break;
                default:
                    break;
            }
        }

//        //断开主进程与子进程的TCP连接
//        SocketServerClose();


        //先执行loc卷积，后执行conf卷积
        for (int i = 0; i < 6; i++) {
            //loc conv
            cnn_layer *layer = &model->layers[i * 2 + 35];

            t0 = get_timestamp();
            loc_result[i] = layer->doFeedForward(loc_franme[i], layer);

            //反量化 转化为全精度帧
            if(layer->useINT8)
            {
                loc_result[i]->data = dequantize_arm((int*)loc_result[i]->data_s8,loc_result[i]->c,loc_result[i]->h,loc_result[i]->w,layer->conv_layer->bias,layer->Scale_in,layer->conv_layer->W_Scale);
                loc_result[i]->useINT8 = 0;
            }

            t1 = get_timestamp();
            double milsecs = (t1 - t0) / 1000.0L;
            LOGI("Processed layer %d in %f ms\n", (i * 2 + 36), milsecs);

            //conf conv
            layer = &model->layers[i * 2 + 36];

            t0 = get_timestamp();
            conf_result[i] = layer->doFeedForward(conf_franme[i], layer);

            //反量化 转化为全精度帧
            if(layer->useINT8)
            {
                conf_result[i]->data = dequantize_arm((int*)conf_result[i]->data_s8,conf_result[i]->c,conf_result[i]->h,conf_result[i]->w,layer->conv_layer->bias,layer->Scale_in,layer->conv_layer->W_Scale);
                conf_result[i]->useINT8 = 0;
            }

            t1 = get_timestamp();
            milsecs = (t1 - t0) / 1000.0L;
            LOGI("Processed layer %d in %f ms\n", (i * 2 + 37), milsecs);

        }

        //检测输出框  最后一层
        cnn_layer *layer = &model->layers[47];

        t0 = get_timestamp();
        layer->num_threads=4;
        result = layer->detectOutputForward(loc_result,conf_result,layer);

        t1 = get_timestamp();
        double milsecs = (t1 - t0) / 1000.0L;
        LOGI("Processed layer 48 in %f ms\n", milsecs);

    }


//    //释放共享内存
//    close_shared_memory(shm_id1,shm_buf1);
//    close_shared_memory(shm_id2,shm_buf2);

    timestamp_t global_t1 = get_timestamp();
    totalTime = (global_t1 - global_t0) / 1000.0L / total_count;
    LOGI("CNN finished in %f ms\n", totalTime);


    for (int i = 0; i < 6; ++i) {
        if (loc_result[i] != NULL) {
            if(model->useGPU)
            {
                frame_free_not_align(loc_result[i]);
            } else
            {
                frame_free(loc_result[i]);
            }
        }
        if (conf_result[i] != NULL) {
            if(model->useGPU)
            {
                frame_free_not_align(conf_result[i]);
            } else
            {
                frame_free(conf_result[i]);
            }
        }
    }

    LOGI("cnn_doClassification over!");

    return result;
}