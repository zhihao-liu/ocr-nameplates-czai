//
// Created by Zhihao Liu on 4/19/18.
//

#ifndef CUIZHOU_OCR_HASHER_H
#define CUIZHOU_OCR_HASHER_H

#include <cassert>
#include <type_traits>
#include <unordered_map>


namespace cz {

template<typename EnumClass>
struct EnumHasher {
    static_assert(std::is_enum<EnumClass>::value, "Struct 'EnumHasher' only supports enum types as template parameters.");

    int operator()(EnumClass enumElem) const;
};

template<typename EnumClass, typename Val>
using EnumHashMap = std::unordered_map<EnumClass, Val, EnumHasher<EnumClass>>;

} // end namespace cz


#include "./impl/enum_hashmap.impl.hpp"

#endif //CUIZHOU_OCR_HASHER_H
