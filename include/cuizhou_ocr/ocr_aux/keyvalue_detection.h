//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <opencv2/core/core.hpp>
#include "ocr_detection.h"


namespace cz {

struct KeyValueDetection {
    OcrDetection key;
    OcrDetection value;

    ~KeyValueDetection();
    KeyValueDetection();

    KeyValueDetection(OcrDetection _key, OcrDetection _value);

    friend std::ostream& operator<<(std::ostream& strm, KeyValueDetection const& obj);
};

} // end namespace cz


#endif //OCR_CUIZHOU_INFOTABLE_H
