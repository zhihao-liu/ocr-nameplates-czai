//
// Created by Zhihao Liu on 4/27/18.
//

#include "datamodel/ocr_detection.h"


namespace cuizhou {

OcrDetection::~OcrDetection() = default;

OcrDetection::OcrDetection() = default;

OcrDetection::OcrDetection(std::string _text, cv::Rect const& _rect)
: text(std::move(_text)), rect(_rect) {};

bool OcrDetection::empty() const {
    return text.empty();
}

} // end namespace cuizhou
