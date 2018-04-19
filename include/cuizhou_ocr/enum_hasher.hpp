//
// Created by Zhihao Liu on 4/19/18.
//

#ifndef CUIZHOU_OCR_HASHER_H
#define CUIZHOU_OCR_HASHER_H

#include <cassert>
#include <type_traits>


namespace cuizhou {

template<typename EnumType>
struct EnumHasher {
    static_assert(std::is_enum<EnumType>::value, "Struct \"EnumHasher\" only supports enum types as template parameters.");
    size_t operator() (EnumType enumElem) const {
        typedef typename std::underlying_type<EnumType>::type EnumData;
        return std::hash<EnumData>()(static_cast<EnumData>(enumElem));
    }
};

} // end namespace cuizhou


#endif //CUIZHOU_OCR_HASHER_H
