//
// Created by Zhihao Liu on 18-4-4.
//

#include "info_table.hpp"


namespace cuizhou {

DetectedItem::DetectedItem(std::string const& _content, cv::Rect const& _rect)
        : content(_content), rect(_rect) {}

KeyValuePair::KeyValuePair(DetectedItem const& _key, DetectedItem const& _value)
        : key(_key), value(_value) {}

void InfoTable::put(std::string const& keyName, KeyValuePair const& keyValuePair) {
    _table.emplace(keyName, keyValuePair);
}

KeyValuePair const* InfoTable::get(std::string const& keyName) const{
    auto itr = _table.find(keyName);
    return itr == _table.end() ? nullptr : &itr->second;
}

void InfoTable::clear() {
    _table.clear();
}

} // end namespace cuizhou
