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
#include "enum_hashmap.hpp"


namespace cuizhou {

template<typename EnumType>
class ClassnameDict {
    static_assert(std::is_enum<EnumType>::value, "Template class \"ClassnameDict\" only support enum types as template parameters!");
public:
    ~ClassnameDict() = default;
    ClassnameDict() = default;
    ClassnameDict(std::vector<EnumType> const& enums, EnumType fallbackEnum,
                  std::vector<std::string> const& names, std::string const& fallbackName,
                  std::vector<std::string> const& aliases = std::vector<std::string>(), std::string const& fallbackAlias = "");

    EnumType toEnum(std::string const& name) const;
    std::string getName(EnumType enumElem) const;
    std::string getAlias(EnumType enumElem) const;
private:
    std::unordered_map< EnumType, std::string, EnumHasher<EnumType> > enumToName_;
    std::unordered_map<std::string, EnumType> nameToEnum_;
    std::unordered_map< EnumType, std::string, EnumHasher<EnumType> > enumToAlias_;

    EnumType const fallbackEnum_;
    std::string const fallbackName_;
    std::string const fallbackAlias_;
};

template<typename EnumType>
ClassnameDict<EnumType>::ClassnameDict(std::vector<EnumType> const& enums, EnumType fallbackEnum,
                                       std::vector<std::string> const& names, std::string const& fallbackName,
                                       std::vector<std::string> const& aliases, std::string const& fallbackAlias)
        : fallbackEnum_(fallbackEnum),
          fallbackName_(fallbackName),
          fallbackAlias_(fallbackAlias) {
    assert(enums.size() == names.size());
    if (!aliases.empty()) assert(enums.size() == aliases.size());
    for (int i = 0; i < enums.size(); ++i) {
        enumToName_.emplace(enums[i], names[i]);
        nameToEnum_.emplace(names[i], enums[i]);
        if (!aliases.empty()) enumToAlias_.emplace(enums[i], aliases[i]);
    }
}

template<typename EnumType>
EnumType ClassnameDict<EnumType>::toEnum(std::string const& name) const {
    auto itr = nameToEnum_.find(name);
    return itr == nameToEnum_.end() ? fallbackEnum_ : itr->second;
}

template<typename EnumType>
std::string ClassnameDict<EnumType>::getName(EnumType enumElem) const {
    auto itr = enumToName_.find(enumElem);
    return itr == enumToName_.end() ? fallbackName_ : itr->second;
}

template<typename EnumType>
std::string ClassnameDict<EnumType>::getAlias(EnumType enumElem) const {
    auto itr = enumToAlias_.find(enumElem);
    return itr == enumToAlias_.end() ? fallbackAlias_ : itr->second;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_CLASSNAME_FINDER_H
