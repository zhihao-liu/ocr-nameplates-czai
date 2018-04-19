//
// Created by Zhihao Liu on 18-4-4.
//

#include "key_value_pair.h"


namespace cuizhou {

DetectedItem::DetectedItem(std::string const& _content, cv::Rect const& _rect)
        : content(_content), rect(_rect) {}

KeyValuePair::KeyValuePair(DetectedItem const& _key, DetectedItem const& _value)
        : key(_key), value(_value) {}

std::ostream& operator<< (std::ostream& strm, KeyValuePair const& obj) {
    return strm << obj.key.content << ": " << obj.value.content;
}

} // end namespace cuizhou

