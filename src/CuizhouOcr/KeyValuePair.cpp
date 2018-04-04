//
// Created by Zhihao Liu on 18-4-4.
//

#include "KeyValuePair.h"


using namespace cuizhou;

DetectedItem::DetectedItem() = default;

DetectedItem::DetectedItem(std::string const& inputContent, cv::Rect const& inputRect)
: content(inputContent), rect(inputRect) {}

KeyValuePair::KeyValuePair() = default;

KeyValuePair::KeyValuePair(DetectedItem const& inputKey, DetectedItem const& inputValue)
        : key(inputKey), value(inputValue) {}