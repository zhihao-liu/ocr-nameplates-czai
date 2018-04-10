//
// Created by Zhihao Liu on 18-4-4.
//

#include "InfoTable.hpp"


using namespace cuizhou;

InfoTable::~InfoTable() = default;

InfoTable::InfoTable() = default;

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