//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrHandler.h"


using namespace cuizhou;

OcrHandler::~OcrHandler() = default;

OcrHandler::OcrHandler() = default;

void OcrHandler::setImage(cv::Mat const& image) {
    _image = image.clone();
}