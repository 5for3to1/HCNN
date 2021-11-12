//
// Created by George on 2019/4/29.
//
#include <layers/detection_output.hpp>
#include <basic_functions.hpp>
#include <deepsense_lib.hpp>
#include <deepsense_internal_lib.hpp>
#include <clio.hpp>
#include <vector>
#include <cmath>
//openmp
#include <omp.h>

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right) {
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores) {
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
}

static inline float intersection_area(const BBoxRect& a, const BBoxRect& b) {
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = bboxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        const BBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (int i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        //过滤小框
        if((a.ymax-a.ymin) < 3 || (a.xmax-a.xmin) < 3)
        {
            continue;
        }

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //后来的小框被前面的大框包含
//            if (inter_area / union_area > nms_threshold || inter_area==areas[i])
//            {
//                keep = 0;
//                break;
//            }
//            else if(inter_area==areas[picked[j]])
//            {
//                picked[j]=i;
//                keep = 0;
//                break;
//            }

            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

cnn_frame * doFeedForward_DETECTION_OUTPUT(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);

    int num_class_copy = ((cnn_layer *)layer)->detection_output_layer->num_class;

    //img 300 300
    //layer_index   min_size    max_size    aspect_ratio    w/h
    //22            60                      2               19
    //26            105         150         2   3           10
    //28            150         195         2   3           5
    //30            195         240         2   3           3
    //32            240         285         2   3           2
    //34            285         300         2   3           1

    int anchor_num[6]={3,6,6,6,6,6};
    float anchor_size[6][6][2];
    double var[6][4];

    /*
//    float image_h_w=300;
//    //layer 22
//    anchor_size[0][0][0]=60/image_h_w;anchor_size[0][0][1]=60/image_h_w;
//    anchor_size[0][1][0]=60/image_h_w*mySqrt(2);anchor_size[0][1][1]=60/image_h_w/mySqrt(2);
//    anchor_size[0][2][0]=60/image_h_w/mySqrt(2);anchor_size[0][2][1]=60/image_h_w*mySqrt(2);
//    var[0][0]=0.10000000149;var[0][1]=0.10000000149;var[0][2]=0.20000000298;var[0][3]=0.20000000298;
//    //layer 26
//    anchor_size[1][0][0]=105/image_h_w;anchor_size[1][0][1]=105/image_h_w;
//    anchor_size[1][1][0]=mySqrt(105/image_h_w*150/image_h_w);anchor_size[1][1][1]=mySqrt(105/image_h_w*150/image_h_w);
//    anchor_size[1][2][0]=105/image_h_w*mySqrt(2);anchor_size[1][2][1]=105/image_h_w/mySqrt(2);
//    anchor_size[1][3][0]=105/mySqrt(2);anchor_size[1][3][1]=105/image_h_w*mySqrt(2);
//    anchor_size[1][4][0]=105/image_h_w*mySqrt(3);anchor_size[1][4][1]=105/image_h_w/mySqrt(3);
//    anchor_size[1][5][0]=105/image_h_w/mySqrt(3);anchor_size[1][5][1]=105/image_h_w*mySqrt(3);
//    var[1][0]=0.10000000149;var[1][1]=0.10000000149;var[1][2]=0.20000000298;var[1][3]=0.20000000298;
//    //layer 28
//    anchor_size[2][0][0]=150/image_h_w;anchor_size[2][0][1]=150/image_h_w;
//    anchor_size[2][1][0]=mySqrt(150/image_h_w*195/image_h_w);anchor_size[2][1][1]=mySqrt(150/image_h_w*195/image_h_w);
//    anchor_size[2][2][0]=150/image_h_w*mySqrt(2);anchor_size[2][2][1]=150/image_h_w/mySqrt(2);
//    anchor_size[2][3][0]=150/image_h_w/mySqrt(2);anchor_size[2][3][1]=150/image_h_w*mySqrt(2);
//    anchor_size[2][4][0]=150/image_h_w*mySqrt(3);anchor_size[2][4][1]=150/image_h_w/mySqrt(3);
//    anchor_size[2][5][0]=150/image_h_w/mySqrt(3);anchor_size[2][5][1]=150/image_h_w*mySqrt(3);
//    var[2][0]=0.10000000149;var[2][1]=0.10000000149;var[2][2]=0.20000000298;var[2][3]=0.20000000298;
//    //layer 30
//    anchor_size[3][0][0]=195/image_h_w;anchor_size[3][0][1]=195/image_h_w;
//    anchor_size[3][1][0]=mySqrt(195/image_h_w*240/image_h_w);anchor_size[3][1][1]=mySqrt(195/image_h_w*240/image_h_w);
//    anchor_size[3][2][0]=195/image_h_w*mySqrt(2);anchor_size[3][2][1]=195/image_h_w/mySqrt(2);
//    anchor_size[3][3][0]=195/image_h_w/mySqrt(2);anchor_size[3][3][1]=195/image_h_w*mySqrt(2);
//    anchor_size[3][4][0]=195/image_h_w*mySqrt(3);anchor_size[3][4][1]=195/image_h_w/mySqrt(3);
//    anchor_size[3][5][0]=195/image_h_w/mySqrt(3);anchor_size[3][5][1]=195/image_h_w*mySqrt(3);
//    var[3][0]=0.10000000149;var[3][1]=0.10000000149;var[3][2]=0.20000000298;var[3][3]=0.20000000298;
//    //layer 32
//    anchor_size[4][0][0]=240/image_h_w;anchor_size[4][0][1]=240/image_h_w;
//    anchor_size[4][1][0]=mySqrt(240/image_h_w*285/image_h_w);anchor_size[4][1][1]=mySqrt(240/image_h_w*285/image_h_w);
//    anchor_size[4][2][0]=240/image_h_w*mySqrt(2);anchor_size[4][2][1]=240/image_h_w/mySqrt(2);
//    anchor_size[4][3][0]=240/image_h_w/mySqrt(2);anchor_size[4][3][1]=240/image_h_w*mySqrt(2);
//    anchor_size[4][4][0]=240/image_h_w*mySqrt(3);anchor_size[4][4][1]=240/image_h_w/mySqrt(3);
//    anchor_size[4][5][0]=240/image_h_w/mySqrt(3);anchor_size[4][5][1]=240/image_h_w*mySqrt(3);
//    var[4][0]=0.10000000149;var[4][1]=0.10000000149;var[4][2]=0.20000000298;var[4][3]=0.20000000298;
//    //layer 34
//    anchor_size[5][0][0]=285/image_h_w;anchor_size[5][0][1]=285/image_h_w;
//    anchor_size[5][1][0]=mySqrt(285/image_h_w*300/image_h_w);anchor_size[5][1][1]=mySqrt(285/image_h_w*300/image_h_w);
//    anchor_size[5][2][0]=285/image_h_w/image_h_w*mySqrt(2);anchor_size[5][2][1]=285/image_h_w/mySqrt(2);
//    anchor_size[5][3][0]=285/image_h_w/mySqrt(2);anchor_size[5][3][1]=285/image_h_w*mySqrt(2);
//    anchor_size[5][4][0]=285/image_h_w*mySqrt(3);anchor_size[5][4][1]=285/image_h_w/mySqrt(3);
//    anchor_size[5][5][0]=285/image_h_w/mySqrt(3);anchor_size[5][5][1]=285/image_h_w*mySqrt(3);
//    var[5][0]=0.10000000149;var[5][1]=0.10000000149;var[5][2]=0.20000000298;var[5][3]=0.20000000298;
    */

    //layer 22
    anchor_size[0][0][0]=60;anchor_size[0][0][1]=60;
    anchor_size[0][1][0]=60*mySqrt(2);anchor_size[0][1][1]=60/mySqrt(2);
    anchor_size[0][2][0]=60/mySqrt(2);anchor_size[0][2][1]=60*mySqrt(2);
    var[0][0]=0.10000000149;var[0][1]=0.10000000149;var[0][2]=0.20000000298;var[0][3]=0.20000000298;
    //layer 26
    anchor_size[1][0][0]=105;anchor_size[1][0][1]=105;
    anchor_size[1][1][0]=mySqrt(105*150);anchor_size[1][1][1]=mySqrt(105*150);
    anchor_size[1][2][0]=105*mySqrt(2);anchor_size[1][2][1]=105/mySqrt(2);
    anchor_size[1][3][0]=105/mySqrt(2);anchor_size[1][3][1]=105*mySqrt(2);
    anchor_size[1][4][0]=105*mySqrt(3);anchor_size[1][4][1]=105/mySqrt(3);
    anchor_size[1][5][0]=105/mySqrt(3);anchor_size[1][5][1]=105*mySqrt(3);
    var[1][0]=0.10000000149;var[1][1]=0.10000000149;var[1][2]=0.20000000298;var[1][3]=0.20000000298;
    //layer 28
    anchor_size[2][0][0]=150;anchor_size[2][0][1]=150;
    anchor_size[2][1][0]=mySqrt(150*195);anchor_size[2][1][1]=mySqrt(150*195);
    anchor_size[2][2][0]=150*mySqrt(2);anchor_size[2][2][1]=150/mySqrt(2);
    anchor_size[2][3][0]=150/mySqrt(2);anchor_size[2][3][1]=150*mySqrt(2);
    anchor_size[2][4][0]=150*mySqrt(3);anchor_size[2][4][1]=150/mySqrt(3);
    anchor_size[2][5][0]=150/mySqrt(3);anchor_size[2][5][1]=150*mySqrt(3);
    var[2][0]=0.10000000149;var[2][1]=0.10000000149;var[2][2]=0.20000000298;var[2][3]=0.20000000298;
    //layer 30
    anchor_size[3][0][0]=195;anchor_size[3][0][1]=195;
    anchor_size[3][1][0]=mySqrt(195*240);anchor_size[3][1][1]=mySqrt(195*240);
    anchor_size[3][2][0]=195*mySqrt(2);anchor_size[3][2][1]=195/mySqrt(2);
    anchor_size[3][3][0]=195/mySqrt(2);anchor_size[3][3][1]=195*mySqrt(2);
    anchor_size[3][4][0]=195*mySqrt(3);anchor_size[3][4][1]=195/mySqrt(3);
    anchor_size[3][5][0]=195/mySqrt(3);anchor_size[3][5][1]=195*mySqrt(3);
    var[3][0]=0.10000000149;var[3][1]=0.10000000149;var[3][2]=0.20000000298;var[3][3]=0.20000000298;
    //layer 32
    anchor_size[4][0][0]=240;anchor_size[4][0][1]=240;
    anchor_size[4][1][0]=mySqrt(240*285);anchor_size[4][1][1]=mySqrt(240*285);
    anchor_size[4][2][0]=240*mySqrt(2);anchor_size[4][2][1]=240/mySqrt(2);
    anchor_size[4][3][0]=240/mySqrt(2);anchor_size[4][3][1]=240*mySqrt(2);
    anchor_size[4][4][0]=240*mySqrt(3);anchor_size[4][4][1]=240/mySqrt(3);
    anchor_size[4][5][0]=240/mySqrt(3);anchor_size[4][5][1]=240*mySqrt(3);
    var[4][0]=0.10000000149;var[4][1]=0.10000000149;var[4][2]=0.20000000298;var[4][3]=0.20000000298;
    //layer 34
    anchor_size[5][0][0]=285;anchor_size[5][0][1]=285;
    anchor_size[5][1][0]=mySqrt(285*300);anchor_size[5][1][1]=mySqrt(285*300);
    anchor_size[5][2][0]=285*mySqrt(2);anchor_size[5][2][1]=285/mySqrt(2);
    anchor_size[5][3][0]=285/mySqrt(2);anchor_size[5][3][1]=285*mySqrt(2);
    anchor_size[5][4][0]=285*mySqrt(3);anchor_size[5][4][1]=285/mySqrt(3);
    anchor_size[5][5][0]=285/mySqrt(3);anchor_size[5][5][1]=285*mySqrt(3);
    var[5][0]=0.10000000149;var[5][1]=0.10000000149;var[5][2]=0.20000000298;var[5][3]=0.20000000298;

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 0; i < 6; ++i) {
        for (int h = 0; h < loc_frame[i]->h; ++h) {
            for (int w = 0; w < loc_frame[i]->w; ++w) {
                for (int k = 0; k < anchor_num[i]; ++k) {

                    float * loc0=loc_frame[i]->data+(k*4*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc1=loc_frame[i]->data+((k*4+1)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc2=loc_frame[i]->data+((k*4+2)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc3=loc_frame[i]->data+((k*4+3)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;

                    float pb_h=anchor_size[i][k][0];
                    float pb_w=anchor_size[i][k][1];
                    float pb_cx=(w+0.5)*300/loc_frame[i]->w;
                    float pb_cy=(h+0.5)*300/loc_frame[i]->h;

                    float bbox_cx = var[i][0] * loc0[0] * pb_w + pb_cx;
                    float bbox_cy = var[i][1] * loc1[0] * pb_h + pb_cy;
                    float bbox_w = myexp(var[i][2] * loc2[0]) * pb_w;
                    float bbox_h = myexp(var[i][3] * loc3[0]) * pb_h;

                    loc0[0] = (bbox_cx - bbox_w * 0.5f);
                    loc1[0] = (bbox_cy - bbox_h * 0.5f);
                    loc2[0] = (bbox_cx + bbox_w * 0.5f);
                    loc3[0] = (bbox_cy + bbox_h * 0.5f);
                }
            }
        }

        //softmax
        cnn_frame * conf_max=frame_init(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);
        cnn_frame * conf_sum=frame_init(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);
        int aligned_hw = alignSize(conf_frame[i]->w*conf_frame[i]->h* sizeof(float),16)/sizeof(float);
        for (int an = 0; an < anchor_num[i]; ++an) {
            fill_float(conf_max->data + an*aligned_hw,conf_frame[i]->w*conf_frame[i]->h,-1000);
            for (int c = 0; c < num_class_copy; ++c)
            {
                const float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw=0; hw<conf_frame[i]->w*conf_frame[i]->h; hw++)
                {
                    conf_max->data[an*aligned_hw + hw] = std::max(conf_max->data[an*aligned_hw + hw], ptr[hw]);
                }
            }
        }
        for (int an = 0; an < anchor_num[i]; ++an) {
            fill_float(conf_sum->data + an*aligned_hw,conf_frame[i]->w*conf_frame[i]->h,0.f);
            for (int c = 0; c < num_class_copy; ++c) {
                float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw = 0; hw < conf_frame[i]->w * conf_frame[i]->h; hw++) {
                    ptr[hw] = myexp(ptr[hw] - conf_max->data[an*aligned_hw + hw]);
                    conf_sum->data[an*aligned_hw + hw] += ptr[hw];
                }
            }
        }
        for (int an = 0; an < anchor_num[i]; ++an) {
            for (int c = 0; c < num_class_copy; ++c) {
                float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw = 0; hw < conf_frame[i]->w * conf_frame[i]->h; hw++) {
                    ptr[hw] /= conf_sum->data[an*aligned_hw + hw];
                }
            }
        }

        frame_free(conf_max);
        frame_free(conf_sum);
    }

    std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class_copy);
    all_class_bbox_scores.resize(num_class_copy);

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 1; i < num_class_copy; ++i) {

        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for (int j = 0; j < 6; ++j) {
            for (int h = 0; h < conf_frame[j]->h; ++h) {
                for (int w = 0; w < conf_frame[j]->w; ++w) {
                    for (int k = 0; k < anchor_num[j]; ++k) {

                        float score=conf_frame[j]->data[((k*num_class_copy+i)*alignSize(conf_frame[j]->w*conf_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*conf_frame[j]->w + w];
                        if(score > ((cnn_layer*)layer)->detection_output_layer->confidence_threshold)
                        {
                            float * loc0=loc_frame[j]->data+(k*4*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc1=loc_frame[j]->data+((k*4+1)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc2=loc_frame[j]->data+((k*4+2)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc3=loc_frame[j]->data+((k*4+3)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;

                            BBoxRect c={loc0[0],loc1[0],loc2[0],loc3[0],i};
                            class_bbox_rects.push_back(c);
                            class_bbox_scores.push_back(score);
                        }
                    }
                }
            }
        }

        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        int nms_top_k=((cnn_layer*)layer)->detection_output_layer->nms_top_k;
        if (nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, ((cnn_layer*)layer)->detection_output_layer->nms_threshold);
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }

    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class_copy; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    qsort_descent_inplace(bbox_rects, bbox_scores);

    int keep_top_k=((cnn_layer*)layer)->detection_output_layer->keep_top_k;
    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
    }

    int num_detected = bbox_rects.size();
    cnn_frame * detection_result=frame_init_not_align(6,num_detected,1,0);

    for (int i = 0; i < num_detected; ++i) {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float * outptr = detection_result->data+i*6;
        outptr[0]=r.label;
        outptr[1]=score;
        outptr[2]=r.xmin;
        outptr[3]=r.ymin;
        outptr[4]=r.xmax;
        outptr[5]=r.ymax;
    }

    return detection_result;
}

cnn_frame * doFeedForward_DETECTION_OUTPUT_GPU(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);

    int num_class_copy = ((cnn_layer *)layer)->detection_output_layer->num_class;

    //img 300 300
    //layer_index   min_size    max_size    aspect_ratio    w/h
    //22            60                      2               19
    //26            105         150         2   3           10
    //28            150         195         2   3           5
    //30            195         240         2   3           3
    //32            240         285         2   3           2
    //34            285         300         2   3           1

    int anchor_num[6]={3,6,6,6,6,6};
    float anchor_size[6][6][2];
    double var[6][4];

    //layer 22
    anchor_size[0][0][0]=60;anchor_size[0][0][1]=60;
    anchor_size[0][1][0]=60*mySqrt(2);anchor_size[0][1][1]=60/mySqrt(2);
    anchor_size[0][2][0]=60/mySqrt(2);anchor_size[0][2][1]=60*mySqrt(2);
    var[0][0]=0.10000000149;var[0][1]=0.10000000149;var[0][2]=0.20000000298;var[0][3]=0.20000000298;
    //layer 26
    anchor_size[1][0][0]=105;anchor_size[1][0][1]=105;
    anchor_size[1][1][0]=mySqrt(105*150);anchor_size[1][1][1]=mySqrt(105*150);
    anchor_size[1][2][0]=105*mySqrt(2);anchor_size[1][2][1]=105/mySqrt(2);
    anchor_size[1][3][0]=105/mySqrt(2);anchor_size[1][3][1]=105*mySqrt(2);
    anchor_size[1][4][0]=105*mySqrt(3);anchor_size[1][4][1]=105/mySqrt(3);
    anchor_size[1][5][0]=105/mySqrt(3);anchor_size[1][5][1]=105*mySqrt(3);
    var[1][0]=0.10000000149;var[1][1]=0.10000000149;var[1][2]=0.20000000298;var[1][3]=0.20000000298;
    //layer 28
    anchor_size[2][0][0]=150;anchor_size[2][0][1]=150;
    anchor_size[2][1][0]=mySqrt(150*195);anchor_size[2][1][1]=mySqrt(150*195);
    anchor_size[2][2][0]=150*mySqrt(2);anchor_size[2][2][1]=150/mySqrt(2);
    anchor_size[2][3][0]=150/mySqrt(2);anchor_size[2][3][1]=150*mySqrt(2);
    anchor_size[2][4][0]=150*mySqrt(3);anchor_size[2][4][1]=150/mySqrt(3);
    anchor_size[2][5][0]=150/mySqrt(3);anchor_size[2][5][1]=150*mySqrt(3);
    var[2][0]=0.10000000149;var[2][1]=0.10000000149;var[2][2]=0.20000000298;var[2][3]=0.20000000298;
    //layer 30
    anchor_size[3][0][0]=195;anchor_size[3][0][1]=195;
    anchor_size[3][1][0]=mySqrt(195*240);anchor_size[3][1][1]=mySqrt(195*240);
    anchor_size[3][2][0]=195*mySqrt(2);anchor_size[3][2][1]=195/mySqrt(2);
    anchor_size[3][3][0]=195/mySqrt(2);anchor_size[3][3][1]=195*mySqrt(2);
    anchor_size[3][4][0]=195*mySqrt(3);anchor_size[3][4][1]=195/mySqrt(3);
    anchor_size[3][5][0]=195/mySqrt(3);anchor_size[3][5][1]=195*mySqrt(3);
    var[3][0]=0.10000000149;var[3][1]=0.10000000149;var[3][2]=0.20000000298;var[3][3]=0.20000000298;
    //layer 32
    anchor_size[4][0][0]=240;anchor_size[4][0][1]=240;
    anchor_size[4][1][0]=mySqrt(240*285);anchor_size[4][1][1]=mySqrt(240*285);
    anchor_size[4][2][0]=240*mySqrt(2);anchor_size[4][2][1]=240/mySqrt(2);
    anchor_size[4][3][0]=240/mySqrt(2);anchor_size[4][3][1]=240*mySqrt(2);
    anchor_size[4][4][0]=240*mySqrt(3);anchor_size[4][4][1]=240/mySqrt(3);
    anchor_size[4][5][0]=240/mySqrt(3);anchor_size[4][5][1]=240*mySqrt(3);
    var[4][0]=0.10000000149;var[4][1]=0.10000000149;var[4][2]=0.20000000298;var[4][3]=0.20000000298;
    //layer 34
    anchor_size[5][0][0]=285;anchor_size[5][0][1]=285;
    anchor_size[5][1][0]=mySqrt(285*300);anchor_size[5][1][1]=mySqrt(285*300);
    anchor_size[5][2][0]=285*mySqrt(2);anchor_size[5][2][1]=285/mySqrt(2);
    anchor_size[5][3][0]=285/mySqrt(2);anchor_size[5][3][1]=285*mySqrt(2);
    anchor_size[5][4][0]=285*mySqrt(3);anchor_size[5][4][1]=285/mySqrt(3);
    anchor_size[5][5][0]=285/mySqrt(3);anchor_size[5][5][1]=285*mySqrt(3);
    var[5][0]=0.10000000149;var[5][1]=0.10000000149;var[5][2]=0.20000000298;var[5][3]=0.20000000298;


    for (int i = 0; i < 6; ++i) {
        loc_frame[i]=frame_convert_to_cpu_not_align(loc_frame[i]);
        conf_frame[i]=frame_convert_to_cpu_not_align(conf_frame[i]);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int h = 0; h < loc_frame[i]->h; ++h) {
            for (int w = 0; w < loc_frame[i]->w; ++w) {
                for (int k = 0; k < anchor_num[i]; ++k) {

                    float * loc0=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4;
                    float * loc1=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 1;
                    float * loc2=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 2;
                    float * loc3=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 3;

                    float pb_h=anchor_size[i][k][0];
                    float pb_w=anchor_size[i][k][1];
                    float pb_cx=(w+0.5)*300/loc_frame[i]->w;
                    float pb_cy=(h+0.5)*300/loc_frame[i]->h;

                    float bbox_cx = var[i][0] * loc0[0] * pb_w + pb_cx;
                    float bbox_cy = var[i][1] * loc1[0] * pb_h + pb_cy;
                    float bbox_w = myexp(var[i][2] * loc2[0]) * pb_w;
                    float bbox_h = myexp(var[i][3] * loc3[0]) * pb_h;

                    loc0[0] = (bbox_cx - bbox_w * 0.5f);
                    loc1[0] = (bbox_cy - bbox_h * 0.5f);
                    loc2[0] = (bbox_cx + bbox_w * 0.5f);
                    loc3[0] = (bbox_cy + bbox_h * 0.5f);
                }
            }
        }

        //softmax
        //考虑到填充函数的汇编执行  这里要用对齐方式申请
        cnn_frame * conf_max=frame_init(conf_frame[i]->w*conf_frame[i]->h,anchor_num[i],1,0);
        cnn_frame * conf_sum=frame_init(conf_frame[i]->w*conf_frame[i]->h,anchor_num[i],1,0);

//        cnn_frame * conf_max=frame_init_dim2(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);
//        cnn_frame * conf_sum=frame_init_dim2(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);

        fill_float(conf_max->data,conf_max->w*conf_max->h*conf_max->c,-1000);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_max->data[hwa] = std::max(conf_max->data[hwa], conf_frame[i]->data[hwa*num_class_copy+c]);
            }
        }

        fill_float(conf_sum->data,conf_sum->w*conf_sum->h*conf_sum->c,0.f);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_frame[i]->data[hwa*num_class_copy+c] = myexp(conf_frame[i]->data[hwa*num_class_copy+c] - conf_max->data[hwa]);
                conf_sum->data[hwa] += conf_frame[i]->data[hwa*num_class_copy+c];
            }
        }
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_frame[i]->data[hwa*num_class_copy+c] /= conf_sum->data[hwa];
            }
        }
        frame_free(conf_max);
        frame_free(conf_sum);
    }

    std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class_copy);
    all_class_bbox_scores.resize(num_class_copy);

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 1; i < num_class_copy; ++i) {

        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for (int j = 0; j < 6; ++j) {
            for (int h = 0; h < conf_frame[j]->h; ++h) {
                for (int w = 0; w < conf_frame[j]->w; ++w) {
                    for (int k = 0; k < anchor_num[j]; ++k) {

                        float score=conf_frame[j]->data[(h*conf_frame[j]->w + w)*conf_frame[j]->c + k*num_class_copy + i];
                        if(score > ((cnn_layer*)layer)->detection_output_layer->confidence_threshold)
                        {
                            float * loc0=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4;
                            float * loc1=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 1;
                            float * loc2=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 2;
                            float * loc3=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 3;

                            BBoxRect c={loc0[0],loc1[0],loc2[0],loc3[0],i};
                            class_bbox_rects.push_back(c);
                            class_bbox_scores.push_back(score);
                        }
                    }
                }
            }
        }

        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        int nms_top_k=((cnn_layer*)layer)->detection_output_layer->nms_top_k;
        if (nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, ((cnn_layer*)layer)->detection_output_layer->nms_threshold);
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }

    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class_copy; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    qsort_descent_inplace(bbox_rects, bbox_scores);

    int keep_top_k=((cnn_layer*)layer)->detection_output_layer->keep_top_k;
    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
    }

    int num_detected = bbox_rects.size();

    cnn_frame * detection_result=frame_init_not_align(6,num_detected,1,0);

    for (int i = 0; i < num_detected; ++i) {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float * outptr = detection_result->data+i*6;
        outptr[0]=r.label;
        outptr[1]=score;
        outptr[2]=r.xmin;
        outptr[3]=r.ymin;
        outptr[4]=r.xmax;
        outptr[5]=r.ymax;
    }

    return detection_result;
}

cnn_frame * doFeedForward_DETECTION_OUTPUT_FACE(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);

    int num_class_copy = ((cnn_layer *)layer)->detection_output_layer->num_class;

    //img 300 300
    //layer_index   min_size    max_size    aspect_ratio    w/h
    //22            16          32          2               19
    //26            32          64          2               10
    //28            64          128         2               5
    //30            128         214         2               3
    //32            214         300         2               2
    //34            214         300         2               1

    int anchor_num[6]={2,2,2,2,2,2};
    float anchor_size[6][2][2];
    double var[6][4];


    //layer 22
    anchor_size[0][0][0]=16;anchor_size[0][0][1]=16;
    anchor_size[0][1][0]=mySqrt(16*32);anchor_size[0][1][1]=mySqrt(16*32);
    var[0][0]=0.10000000149;var[0][1]=0.10000000149;var[0][2]=0.20000000298;var[0][3]=0.20000000298;
    //layer 26
    anchor_size[1][0][0]=32;anchor_size[1][0][1]=32;
    anchor_size[1][1][0]=mySqrt(32*64);anchor_size[1][1][1]=mySqrt(32*64);
    var[1][0]=0.10000000149;var[1][1]=0.10000000149;var[1][2]=0.20000000298;var[1][3]=0.20000000298;
    //layer 28
    anchor_size[2][0][0]=64;anchor_size[2][0][1]=64;
    anchor_size[2][1][0]=mySqrt(64*128);anchor_size[2][1][1]=mySqrt(64*128);
    var[2][0]=0.10000000149;var[2][1]=0.10000000149;var[2][2]=0.20000000298;var[2][3]=0.20000000298;
    //layer 30
    anchor_size[3][0][0]=128;anchor_size[3][0][1]=128;
    anchor_size[3][1][0]=mySqrt(128*214);anchor_size[3][1][1]=mySqrt(128*214);
    var[3][0]=0.10000000149;var[3][1]=0.10000000149;var[3][2]=0.20000000298;var[3][3]=0.20000000298;
    //layer 32
    anchor_size[4][0][0]=214;anchor_size[4][0][1]=214;
    anchor_size[4][1][0]=mySqrt(240*300);anchor_size[4][1][1]=mySqrt(214*300);
    var[4][0]=0.10000000149;var[4][1]=0.10000000149;var[4][2]=0.20000000298;var[4][3]=0.20000000298;
    //layer 34
    anchor_size[5][0][0]=214;anchor_size[5][0][1]=214;
    anchor_size[5][1][0]=mySqrt(214*300);anchor_size[5][1][1]=mySqrt(214*300);
    var[5][0]=0.10000000149;var[5][1]=0.10000000149;var[5][2]=0.20000000298;var[5][3]=0.20000000298;

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 0; i < 6; ++i) {
        for (int h = 0; h < loc_frame[i]->h; ++h) {
            for (int w = 0; w < loc_frame[i]->w; ++w) {
                for (int k = 0; k < anchor_num[i]; ++k) {

                    float * loc0=loc_frame[i]->data+(k*4*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc1=loc_frame[i]->data+((k*4+1)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc2=loc_frame[i]->data+((k*4+2)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;
                    float * loc3=loc_frame[i]->data+((k*4+3)*alignSize(loc_frame[i]->w*loc_frame[i]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[i]->w + w;

                    float pb_h=anchor_size[i][k][0];
                    float pb_w=anchor_size[i][k][1];
                    float pb_cx=(w+0.5)*300/loc_frame[i]->w;
                    float pb_cy=(h+0.5)*300/loc_frame[i]->h;

                    float bbox_cx = var[i][0] * loc0[0] * pb_w + pb_cx;
                    float bbox_cy = var[i][1] * loc1[0] * pb_h + pb_cy;
                    float bbox_w = myexp(var[i][2] * loc2[0]) * pb_w;
                    float bbox_h = myexp(var[i][3] * loc3[0]) * pb_h;

                    loc0[0] = (bbox_cx - bbox_w * 0.5f);
                    loc1[0] = (bbox_cy - bbox_h * 0.5f);
                    loc2[0] = (bbox_cx + bbox_w * 0.5f);
                    loc3[0] = (bbox_cy + bbox_h * 0.5f);
                }
            }
        }

        //softmax
        cnn_frame * conf_max=frame_init(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);
        cnn_frame * conf_sum=frame_init(conf_frame[i]->w,conf_frame[i]->h,anchor_num[i],0);
        int aligned_hw = alignSize(conf_frame[i]->w*conf_frame[i]->h* sizeof(float),16)/sizeof(float);
        for (int an = 0; an < anchor_num[i]; ++an) {
            fill_float(conf_max->data + an*aligned_hw,conf_frame[i]->w*conf_frame[i]->h,-1000);
            for (int c = 0; c < num_class_copy; ++c)
            {
                const float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw=0; hw<conf_frame[i]->w*conf_frame[i]->h; hw++)
                {
                    conf_max->data[an*aligned_hw + hw] = std::max(conf_max->data[an*aligned_hw + hw], ptr[hw]);
                }
            }
        }
        for (int an = 0; an < anchor_num[i]; ++an) {
            fill_float(conf_sum->data + an*aligned_hw,conf_frame[i]->w*conf_frame[i]->h,0.f);
            for (int c = 0; c < num_class_copy; ++c) {
                float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw = 0; hw < conf_frame[i]->w * conf_frame[i]->h; hw++) {
                    ptr[hw] = myexp(ptr[hw] - conf_max->data[an*aligned_hw + hw]);
                    conf_sum->data[an*aligned_hw + hw] += ptr[hw];
                }
            }
        }
        for (int an = 0; an < anchor_num[i]; ++an) {
            for (int c = 0; c < num_class_copy; ++c) {
                float* ptr = conf_frame[i]->data + (an*num_class_copy+c)*aligned_hw;
                for (int hw = 0; hw < conf_frame[i]->w * conf_frame[i]->h; hw++) {
                    ptr[hw] /= conf_sum->data[an*aligned_hw + hw];
                }
            }
        }

        frame_free(conf_max);
        frame_free(conf_sum);
    }

    std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class_copy);
    all_class_bbox_scores.resize(num_class_copy);

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 1; i < num_class_copy; ++i) {

        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for (int j = 0; j < 6; ++j) {
            for (int h = 0; h < conf_frame[j]->h; ++h) {
                for (int w = 0; w < conf_frame[j]->w; ++w) {
                    for (int k = 0; k < anchor_num[j]; ++k) {

                        float score=conf_frame[j]->data[((k*num_class_copy+i)*alignSize(conf_frame[j]->w*conf_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*conf_frame[j]->w + w];
                        if(score > ((cnn_layer*)layer)->detection_output_layer->confidence_threshold)
                        {
                            float * loc0=loc_frame[j]->data+(k*4*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc1=loc_frame[j]->data+((k*4+1)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc2=loc_frame[j]->data+((k*4+2)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;
                            float * loc3=loc_frame[j]->data+((k*4+3)*alignSize(loc_frame[j]->w*loc_frame[j]->h* sizeof(float),16)/sizeof(float)) + h*loc_frame[j]->w + w;

                            BBoxRect c={loc0[0],loc1[0],loc2[0],loc3[0],i};
                            class_bbox_rects.push_back(c);
                            class_bbox_scores.push_back(score);
                        }
                    }
                }
            }
        }

        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        int nms_top_k=((cnn_layer*)layer)->detection_output_layer->nms_top_k;
        if (nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, ((cnn_layer*)layer)->detection_output_layer->nms_threshold);
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }

    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class_copy; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    qsort_descent_inplace(bbox_rects, bbox_scores);

    int keep_top_k=((cnn_layer*)layer)->detection_output_layer->keep_top_k;
    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
    }

    int num_detected = bbox_rects.size();
    cnn_frame * detection_result=frame_init_not_align(6,num_detected,1,0);

    for (int i = 0; i < num_detected; ++i) {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float * outptr = detection_result->data+i*6;
        outptr[0]=r.label;
        outptr[1]=score;
        outptr[2]=r.xmin;
        outptr[3]=r.ymin;
        outptr[4]=r.xmax;
        outptr[5]=r.ymax;
    }

    return detection_result;
}

cnn_frame * doFeedForward_DETECTION_OUTPUT_FACE_GPU(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer) {

    LOGI("Running function %s", __PRETTY_FUNCTION__);

    int num_class_copy = ((cnn_layer *)layer)->detection_output_layer->num_class;

    //img 300 300
    //layer_index   min_size    max_size    aspect_ratio    w/h
    //22            16          32          2               19
    //26            32          64          2               10
    //28            64          128         2               5
    //30            128         214         2               3
    //32            214         300         2               2
    //34            214         300         2               1

    int anchor_num[6]={2,2,2,2,2,2};
    float anchor_size[6][2][2];
    double var[6][4];

    //layer 22
    anchor_size[0][0][0]=16;anchor_size[0][0][1]=16;
    anchor_size[0][1][0]=mySqrt(16*32);anchor_size[0][1][1]=mySqrt(16*32);
    var[0][0]=0.10000000149;var[0][1]=0.10000000149;var[0][2]=0.20000000298;var[0][3]=0.20000000298;
    //layer 26
    anchor_size[1][0][0]=32;anchor_size[1][0][1]=32;
    anchor_size[1][1][0]=mySqrt(32*64);anchor_size[1][1][1]=mySqrt(32*64);
    var[1][0]=0.10000000149;var[1][1]=0.10000000149;var[1][2]=0.20000000298;var[1][3]=0.20000000298;
    //layer 28
    anchor_size[2][0][0]=64;anchor_size[2][0][1]=64;
    anchor_size[2][1][0]=mySqrt(64*128);anchor_size[2][1][1]=mySqrt(64*128);
    var[2][0]=0.10000000149;var[2][1]=0.10000000149;var[2][2]=0.20000000298;var[2][3]=0.20000000298;
    //layer 30
    anchor_size[3][0][0]=128;anchor_size[3][0][1]=128;
    anchor_size[3][1][0]=mySqrt(128*214);anchor_size[3][1][1]=mySqrt(128*214);
    var[3][0]=0.10000000149;var[3][1]=0.10000000149;var[3][2]=0.20000000298;var[3][3]=0.20000000298;
    //layer 32
    anchor_size[4][0][0]=214;anchor_size[4][0][1]=214;
    anchor_size[4][1][0]=mySqrt(240*300);anchor_size[4][1][1]=mySqrt(214*300);
    var[4][0]=0.10000000149;var[4][1]=0.10000000149;var[4][2]=0.20000000298;var[4][3]=0.20000000298;
    //layer 34
    anchor_size[5][0][0]=214;anchor_size[5][0][1]=214;
    anchor_size[5][1][0]=mySqrt(214*300);anchor_size[5][1][1]=mySqrt(214*300);
    var[5][0]=0.10000000149;var[5][1]=0.10000000149;var[5][2]=0.20000000298;var[5][3]=0.20000000298;


    for (int i = 0; i < 6; ++i) {
        loc_frame[i]=frame_convert_to_cpu_not_align(loc_frame[i]);
        conf_frame[i]=frame_convert_to_cpu_not_align(conf_frame[i]);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int h = 0; h < loc_frame[i]->h; ++h) {
            for (int w = 0; w < loc_frame[i]->w; ++w) {
                for (int k = 0; k < anchor_num[i]; ++k) {

                    float * loc0=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4;
                    float * loc1=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 1;
                    float * loc2=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 2;
                    float * loc3=loc_frame[i]->data + (h*loc_frame[i]->w + w)*loc_frame[i]->c + k*4 + 3;

                    float pb_h=anchor_size[i][k][0];
                    float pb_w=anchor_size[i][k][1];
                    float pb_cx=(w+0.5)*300/loc_frame[i]->w;
                    float pb_cy=(h+0.5)*300/loc_frame[i]->h;

                    float bbox_cx = var[i][0] * loc0[0] * pb_w + pb_cx;
                    float bbox_cy = var[i][1] * loc1[0] * pb_h + pb_cy;
                    float bbox_w = myexp(var[i][2] * loc2[0]) * pb_w;
                    float bbox_h = myexp(var[i][3] * loc3[0]) * pb_h;

                    loc0[0] = (bbox_cx - bbox_w * 0.5f);
                    loc1[0] = (bbox_cy - bbox_h * 0.5f);
                    loc2[0] = (bbox_cx + bbox_w * 0.5f);
                    loc3[0] = (bbox_cy + bbox_h * 0.5f);
                }
            }
        }

        //softmax
        //考虑到填充函数的汇编执行  这里要用对齐方式申请
        cnn_frame * conf_max=frame_init(conf_frame[i]->w*conf_frame[i]->h,anchor_num[i],1,0);
        cnn_frame * conf_sum=frame_init(conf_frame[i]->w*conf_frame[i]->h,anchor_num[i],1,0);

        fill_float(conf_max->data,conf_max->w*conf_max->h*conf_max->c,-1000);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_max->data[hwa] = std::max(conf_max->data[hwa], conf_frame[i]->data[hwa*num_class_copy+c]);
            }
        }

        fill_float(conf_sum->data,conf_sum->w*conf_sum->h*conf_sum->c,0.f);
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_frame[i]->data[hwa*num_class_copy+c] = myexp(conf_frame[i]->data[hwa*num_class_copy+c] - conf_max->data[hwa]);
                conf_sum->data[hwa] += conf_frame[i]->data[hwa*num_class_copy+c];
            }
        }
#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
        for (int hwa=0; hwa<conf_frame[i]->w*conf_frame[i]->h*anchor_num[i]; hwa++)
        {
            for (int c = 0; c < num_class_copy; ++c)
            {
                conf_frame[i]->data[hwa*num_class_copy+c] /= conf_sum->data[hwa];
            }
        }
        frame_free(conf_max);
        frame_free(conf_sum);
    }

    std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class_copy);
    all_class_bbox_scores.resize(num_class_copy);

#pragma omp parallel for num_threads(((cnn_layer *)layer)->num_threads)
    for (int i = 1; i < num_class_copy; ++i) {

        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for (int j = 0; j < 6; ++j) {
            for (int h = 0; h < conf_frame[j]->h; ++h) {
                for (int w = 0; w < conf_frame[j]->w; ++w) {
                    for (int k = 0; k < anchor_num[j]; ++k) {

                        float score=conf_frame[j]->data[(h*conf_frame[j]->w + w)*conf_frame[j]->c + k*num_class_copy + i];
                        if(score > ((cnn_layer*)layer)->detection_output_layer->confidence_threshold)
                        {
                            float * loc0=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4;
                            float * loc1=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 1;
                            float * loc2=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 2;
                            float * loc3=loc_frame[j]->data + (h*loc_frame[j]->w + w) * loc_frame[j]->c + k*4 + 3;

                            BBoxRect c={loc0[0],loc1[0],loc2[0],loc3[0],i};
                            class_bbox_rects.push_back(c);
                            class_bbox_scores.push_back(score);
                        }
                    }
                }
            }
        }

        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        int nms_top_k=((cnn_layer*)layer)->detection_output_layer->nms_top_k;
        if (nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, ((cnn_layer*)layer)->detection_output_layer->nms_threshold);
        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }

    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class_copy; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    qsort_descent_inplace(bbox_rects, bbox_scores);

    int keep_top_k=((cnn_layer*)layer)->detection_output_layer->keep_top_k;
    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
    }

    int num_detected = bbox_rects.size();

    cnn_frame * detection_result=frame_init_not_align(6,num_detected,1,0);

    for (int i = 0; i < num_detected; ++i) {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float * outptr = detection_result->data+i*6;
        outptr[0]=r.label;
        outptr[1]=score;
        outptr[2]=r.xmin;
        outptr[3]=r.ymin;
        outptr[4]=r.xmax;
        outptr[5]=r.ymax;
    }

    return detection_result;
}
