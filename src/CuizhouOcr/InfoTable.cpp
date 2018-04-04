//
// Created by Zhihao Liu on 18-4-4.
//

#include "InfoTable.h"


using namespace cuizhou;

InfoTable::~InfoTable() = default;

InfoTable::InfoTable() = default;

void InfoTable::put(std::string const& keyName, KeyValuePair const& keyValuePair) {
    _table.insert(std::make_pair(keyName, keyValuePair));
}

KeyValuePair const& InfoTable::get(std::string const& keyName) {
    return _table[keyName];
}