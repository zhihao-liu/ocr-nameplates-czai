//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <opencv2/core/core.hpp>
#include "data_aux/ocr_detection.h"


namespace cuizhou {

struct KeyValueDetection {
    OcrDetection key;
    OcrDetection value;

    ~KeyValueDetection() = default;
    KeyValueDetection() = default;

    KeyValueDetection(OcrDetection _key, OcrDetection _value)
            : key(std::move(_key)), value(std::move(_value)) {};

    friend std::ostream& operator<<(std::ostream& strm, KeyValueDetection const& obj) {
        return strm << obj.key.text << ": " << obj.value.text;
    }
};

} // end namespace cuizhou

#endif //OCR_CUIZHOU_INFOTABLE_H
