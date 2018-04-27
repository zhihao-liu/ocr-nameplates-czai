//
// Created by Zhihao Liu on 4/27/18.
//

#include "datamodel/keyvalue_detection.h"


namespace cuizhou {

KeyValueDetection::~KeyValueDetection() = default;

KeyValueDetection::KeyValueDetection() = default;

KeyValueDetection::KeyValueDetection(OcrDetection _key, OcrDetection _value)
: key(std::move(_key)), value(std::move(_value)) {};

std::ostream& operator<<(std::ostream& strm, KeyValueDetection const& obj) {
    return strm << obj.key.text << ": " << obj.value.text;
}

} // end namespace cuizhou
