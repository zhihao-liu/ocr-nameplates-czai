//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <opencv2/core/core.hpp>


namespace cuizhou {

class DetectedItem {
public:
    std::string content;
    cv::Rect rect;

    ~DetectedItem() = default;
    DetectedItem() = default;
    DetectedItem(std::string const& _content, cv::Rect const& _rect);
};

class KeyValuePair {
public:
    DetectedItem key;
    DetectedItem value;

    ~KeyValuePair() = default;
    KeyValuePair() = default;
    KeyValuePair(DetectedItem const& _key, DetectedItem const& _value);

    friend std::ostream& operator<< (std::ostream& strm, KeyValuePair const& obj);
};

} // end namespace cuizhou

#endif //OCR_CUIZHOU_INFOTABLE_H
