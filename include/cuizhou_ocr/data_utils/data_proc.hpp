//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_DATA_PROCESSING_H
#define CUIZHOU_OCR_DATA_PROCESSING_H

#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include "enum_hashmap.hpp"


namespace cz {

template<typename Container>
auto findMedian(Container containerCopy) -> typename Container::value_type;
template<typename Container, typename Func>
auto findMedian(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type;

template<typename Container>
auto computeMean(Container const& container) -> typename Container::value_type;
template<typename Container, typename Func>
auto computeMean(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type;

template<typename Container, typename ConflictPred, typename Compare>
void resolveConflicts(Container& container, ConflictPred&& conflict, Compare&& comp);

template<typename Container, typename Field, typename RefObj, typename RelevancePred>
auto distributeItemsByField(Container const& items, EnumHashMap<Field, RefObj> const& refMap, RelevancePred&& relevant) -> EnumHashMap<Field, Container>;

struct LinearFit {
public:
    LinearFit(std::vector<double> const& x, std::vector<double> const& y);
    double slope() const;
    double constant() const;

private:
    double a, b;
};

} // end namespace cz


#include "./impl/data_proc.impl.hpp"

#endif //CUIZHOU_OCR_DATA_PROCESSING_H