//
// Created by Zhihao Liu on 18-4-4.
//

#include "key_value_pair.h"


namespace cuizhou {

DetectedItem::DetectedItem(std::string const& _text, cv::Rect const& _rect)
        : text(_text), rect(_rect) {}

bool DetectedItem::empty() {
    return text.empty();
}

KeyValuePair::KeyValuePair(DetectedItem const& _key, DetectedItem const& _value)
        : key(_key), value(_value) {}

std::ostream& operator<<(std::ostream& strm, KeyValuePair const& obj) {
    return strm << obj.key.text << ": " << obj.value.text;
}

} // end namespace cuizhou

