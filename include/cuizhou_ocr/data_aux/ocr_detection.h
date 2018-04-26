//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_OCR_DETECTION_H
#define CUIZHOU_OCR_OCR_DETECTION_H

namespace cuizhou {

struct OcrDetection {
    std::string text;
    cv::Rect rect;

    ~OcrDetection() = default;
    OcrDetection() = default;

    OcrDetection(std::string _text, cv::Rect _rect)
            : text(std::move(_text)), rect(std::move(_rect)) {};

    bool empty() const { return text.empty(); };
};

} // end namespace cuizhou


#endif //CUIZHOU_OCR_DETECTED_ITEM_H
