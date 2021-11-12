#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include "deepsense_lib.hpp"

cnn_frame * cnn_doClassification(cnn_frame *frame, cnn *model);
#endif
