namespace cuizhou {

template<typename EnumClass>
ClassnameDict<EnumClass>::~ClassnameDict() = default;

template<typename EnumClass>
ClassnameDict<EnumClass>::ClassnameDict() = default;

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