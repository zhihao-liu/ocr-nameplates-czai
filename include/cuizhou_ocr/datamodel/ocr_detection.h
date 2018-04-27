//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_OCR_DETECTION_H
#define CUIZHOU_OCR_OCR_DETECTION_H

#include <string>
#include <opencv2/core/core.hpp>

namespace cuizhou {

struct OcrDetection {
    std::string text;
    cv::Rect rect;

    ~OcrDetection();
    OcrDetection();

    OcrDetection(std::string _text, cv::Rect const& _rect);

    bool empty() const;
};

} // end namespace cuizhou


#endif //CUIZHOU_OCR_DETECTED_ITEM_H
