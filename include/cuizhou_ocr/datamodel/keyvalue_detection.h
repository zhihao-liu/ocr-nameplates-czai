//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <opencv2/core/core.hpp>
#include "datamodel/ocr_detection.h"


namespace cuizhou {

struct KeyValueDetection {
    OcrDetection key;
    OcrDetection value;

    ~KeyValueDetection();
    KeyValueDetection();

    KeyValueDetection(OcrDetection _key, OcrDetection _value);

    friend std::ostream& operator<<(std::ostream& strm, KeyValueDetection const& obj);
};

} // end namespace cuizhou


#endif //OCR_CUIZHOU_INFOTABLE_H
