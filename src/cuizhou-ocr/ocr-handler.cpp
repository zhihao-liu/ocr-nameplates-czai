//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr-handler.h"


namespace cuizhou {

void OcrHandler::setImage(cv::Mat const& image) {
    image_ = image.clone();
}

cv::Mat const& OcrHandler::image() const {
    return image_;
}

} // end namespace cuizhou
