//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_KEYVALUEPAIR_H
#define OCR_CUIZHOU_KEYVALUEPAIR_H

#include "opencv2/core/core.hpp"


namespace cuizhou {
    struct DetectedItem {
        std::string const content;
        cv::Rect const rect;

        DetectedItem();
        DetectedItem(std::string const& inputContent, cv::Rect const& inputRect);
    };

    struct KeyValuePair {
        DetectedItem const key;
        DetectedItem const value;

        KeyValuePair();
        KeyValuePair(DetectedItem const &key, DetectedItem const &value);
    };
}


#endif //OCR_CUIZHOU_KEYVALUEPAIR_H
