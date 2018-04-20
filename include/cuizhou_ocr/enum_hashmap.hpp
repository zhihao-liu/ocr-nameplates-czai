//
// Created by Zhihao Liu on 4/19/18.
//

#ifndef CUIZHOU_OCR_HASHER_H
#define CUIZHOU_OCR_HASHER_H

#include <cassert>
#include <type_traits>
#include <unordered_map>


namespace cuizhou {

template<typename EnumType>
struct EnumHasher {
    static_assert(std::is_enum<EnumType>::value, "Struct \"EnumHasher\" only supports enum types as template parameters.");
    size_t operator() (EnumType enumElem) const {
        using EnumData = typename std::underlying_type<EnumType>::type;
        return std::hash<EnumData>()(static_cast<EnumData>(enumElem));
    }
};

template<typename EnumType, typename ValType>
using EnumHashMap = std::unordered_map< EnumType, ValType, EnumHasher<EnumType> >;

} // end namespace cuizhou


#endif //CUIZHOU_OCR_HASHER_H
