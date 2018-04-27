namespace cuizhou {

template<typename EnumClass>
int EnumHasher<EnumClass>::operator()(EnumClass enumElem) const {
    using EnumData = typename std::underlying_type<EnumClass>::type;
    return std::hash<EnumData>()(static_cast<EnumData>(enumElem));
}

} // end namespace cuizhou