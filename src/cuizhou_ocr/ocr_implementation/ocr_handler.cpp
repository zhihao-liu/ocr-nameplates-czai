//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_implementation/ocr_handler.h"


namespace cuizhou {

OcrHandler::~OcrHandler() = default;

OcrHandler::OcrHandler() = default;

void OcrHandler::importImage(cv::Mat const& image) {
    image_ = image.clone();
}

void OcrHandler::setImageSource(cv::Mat const& image) {
    image_ = image;
}

cv::Mat const& OcrHandler::image() const {
    return image_;
}

} // end namespace cuizhou
