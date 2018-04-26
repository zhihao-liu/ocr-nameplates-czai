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

template<typename EnumClass>
class ClassnameDict {
    static_assert(std::is_enum<EnumClass>::value, "Template class \"ClassnameDict\" only support enum types as template parameters!");
public:
    ~ClassnameDict() = default;
    ClassnameDict() = default;
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

template<typename EnumClass>
ClassnameDict<EnumClass>::ClassnameDict(std::vector<EnumClass> const& enums, EnumClass fallbackEnum,
                                       std::vector<std::string> const& names, std::string fallbackName,
                                       std::vector<std::string> const& aliases, std::string fallbackAlias)
        : fallbackEnum_(fallbackEnum),
          fallbackName_(std::move(fallbackName)),
          fallbackAlias_(std::move(fallbackAlias)) {
    assert(enums.size() == names.size());
    if (!aliases.empty()) assert(enums.size() == aliases.size());
    for (int i = 0; i < enums.size(); ++i) {
        enumToName_.emplace(enums[i], names[i]);
        nameToEnum_.emplace(names[i], enums[i]);
        if (!aliases.empty()) enumToAlias_.emplace(enums[i], aliases[i]);
    }
}

template<typename EnumClass>
EnumClass ClassnameDict<EnumClass>::toEnum(std::string const& name) const {
    auto itr = nameToEnum_.find(name);
    return itr == nameToEnum_.end() ? fallbackEnum_ : itr->second;
}

template<typename EnumClass>
std::string ClassnameDict<EnumClass>::getName(EnumClass enumItem) const {
    auto itr = enumToName_.find(enumItem);
    return itr == enumToName_.end() ? fallbackName_ : itr->second;
}

template<typename EnumClass>
std::string ClassnameDict<EnumClass>::getAlias(EnumClass enumItem) const {
    auto itr = enumToAlias_.find(enumItem);
    return itr == enumToAlias_.end() ? fallbackAlias_ : itr->second;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_CLASSNAME_FINDER_H
