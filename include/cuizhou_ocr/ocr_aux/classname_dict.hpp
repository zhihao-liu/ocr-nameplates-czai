//
// Created by Zhihao Liu on 4/18/18.
//

#ifndef CUIZHOU_OCR_CLASSNAME_FINDER_H
#define CUIZHOU_OCR_CLASSNAME_FINDER_H

#include <type_traits>
#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include "data_utils/enum_hashmap.hpp"


namespace cz {

template<typename EnumClass>
class ClassnameDict {
    static_assert(std::is_enum<EnumClass>::value, "Template class 'ClassnameDict' only support enum types as template parameters!");

public:
    ~ClassnameDict();
    ClassnameDict();

    ClassnameDict(std::vector<EnumClass> const& enums, EnumClass fallbackEnum,
                  std::vector<std::string> const& names, std::string fallbackName,
                  std::vector<std::string> const& aliases = std::vector<std::string>(), std::string fallbackAlias = "");

    EnumClass toEnum(std::string const& name) const;
    std::string getName(EnumClass enumItem) const;
    std::string getAlias(EnumClass enumItem) const;

private:
    std::unordered_map<EnumClass, std::string, EnumHasher<EnumClass>> enumToName_;
    std::unordered_map<std::string, EnumClass> nameToEnum_;
    std::unordered_map<EnumClass, std::string, EnumHasher<EnumClass>> enumToAlias_;

    EnumClass const fallbackEnum_;
    std::string const fallbackName_;
    std::string const fallbackAlias_;
};

} // end namespace cz


#include "./impl/classname_dict.impl.hpp"

#endif //CUIZHOU_OCR_CLASSNAME_FINDER_H
