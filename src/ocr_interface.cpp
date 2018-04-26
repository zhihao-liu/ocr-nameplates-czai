//
// Created by Zhihao Liu on 4/26/18.
//

#include "ocr_interface.hpp"

namespace cuizhou {

void OcrInterface::inputImage(cv::Mat const& img) {
    ocrHandler_->inputImage(img);
}

void OcrInterface::processImage() {
    ocrHandler_->processImage();
}

cv::Mat const& OcrInterface::image() const {
    return ocrHandler_->image();
}

std::string OcrInterface::getResultAsString() const {
    return ocrHandler_->getResultAsString();
}

cv::Mat OcrInterface::drawResult() const {
    return ocrHandler_->drawResult();
}

} // end namespace cuizhou