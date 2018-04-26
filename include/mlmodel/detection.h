#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <opencv2/core/core.hpp>


namespace cuizhou {

struct Detection {
    std::string label;
    cv::Rect rect;
    float score;

    ~Detection() = default;
    Detection() = default;

    Detection(std::string _label, cv::Rect _rect, float _score)
            : label(std::move(_label)), rect(std::move(_rect)), score(_score) {};
};

}

#endif // DETECTION_H
