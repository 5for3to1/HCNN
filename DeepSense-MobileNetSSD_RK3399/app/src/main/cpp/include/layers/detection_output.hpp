//
// Created by George on 2019/4/29.
// 对象检测结果后处理
//

#ifndef DEEPSENSE_MOBILENETSSD_RK3399_DETECTION_OUTPUT_HPP
#define DEEPSENSE_MOBILENETSSD_RK3399_DETECTION_OUTPUT_HPP

#include <deepsense_lib.hpp>

cnn_frame * doFeedForward_DETECTION_OUTPUT(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer);

cnn_frame * doFeedForward_DETECTION_OUTPUT_GPU(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer);

cnn_frame * doFeedForward_DETECTION_OUTPUT_FACE(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer);

cnn_frame * doFeedForward_DETECTION_OUTPUT_FACE_GPU(cnn_frame * loc_frame[],cnn_frame * conf_frame[],void *layer);

#endif //DEEPSENSE_MOBILENETSSD_RK3399_DETECTION_OUTPUT_HPP
