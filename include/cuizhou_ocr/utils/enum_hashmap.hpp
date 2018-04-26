//
// Created by Zhihao Liu on 4/19/18.
//

#ifndef CUIZHOU_OCR_HASHER_H
#define CUIZHOU_OCR_HASHER_H

#include <cassert>
#include <type_traits>
#include <unordered_map>


namespace cuizhou {

template<typename EnumClass>
struct EnumHasher {
    static_assert(std::is_enum<EnumClass>::value, "Struct \"EnumHasher\" only supports enum types as template parameters.");
    size_t operator() (EnumClass enumElem) const {
        using EnumData = typename std::underlying_type<EnumClass>::type;
        return std::hash<EnumData>()(static_cast<EnumData>(enumElem));
    }
};

template<typename EnumClass, typename Val>
using EnumHashMap = std::unordered_map<EnumClass, Val, EnumHasher<EnumClass>>;

} // end namespace cuizhou


#endif //CUIZHOU_OCR_HASHER_H
