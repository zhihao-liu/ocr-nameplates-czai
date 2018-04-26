//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_DATA_PROCESSING_H
#define CUIZHOU_OCR_DATA_PROCESSING_H

#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>


namespace cuizhou {

template<typename Container, typename Conflict, typename Compare>
void resolveConflicts(Container& container, Conflict&& conflict, Compare&& comp);

template<typename Container>
auto findMedian(Container containerCopy) -> typename Container::value_type;
template<typename Container, typename Func>
auto findMedian(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type;

template<typename Container>
auto computeMean(Container const& container) -> typename Container::value_type;
template<typename Container, typename Func>
auto computeMean(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type;

class LinearFit {
public:
    LinearFit(std::vector<double> const& x, std::vector<double> const& y);
    double slope() const { return a; };
    double constant() const { return b; };
private:
    double a, b;
};

/* ------- Template Implementations -------*/

// find the median of a collection of numbers
template<typename Container>
auto findMedian(Container containerCopy) -> typename Container::value_type {
    assert(!containerCopy.empty());

    std::nth_element(containerCopy.begin(), containerCopy.begin() + containerCopy.size() / 2, containerCopy.end());

    return containerCopy.at(containerCopy.size() / 2);
};

// find the median of a collection with a given functor that maps each item to a number
template<typename Container, typename Func>
auto findMedian(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type {
    assert(!container.empty());

    using Item = typename Container::value_type;
    using Result = typename std::result_of<Func(Item)>::type;

    std::vector<Result> numbers;
    std::transform(container.cbegin(), container.cend(), std::back_inserter(numbers), [&](Item const& item) { return toNumber(item); });
    std::nth_element(numbers.begin(), numbers.begin() + numbers.size() / 2, numbers.end());

    return numbers.at(numbers.size() / 2);
};

// compute the mean of a collection of numbers
template<typename Container>
auto computeMean(Container const& container) -> typename Container::value_type {
    assert(!container.empty());

    using Item = typename Container::value_type;

    Item  sum = std::accumulate(container.cbegin(), container.cend(), Item(0));

    return sum / container.size();
};

// compute the mean of a collection with a given functor that maps each item to a number
template<typename Container, typename Func>
auto computeMean(Container const& container, Func&& toNumber) -> typename std::result_of<Func(typename Container::value_type)>::type {
    assert(!container.empty());

    using Item = typename Container::value_type;
    using Result = typename std::result_of<Func(Item)>::type;

    Result sum = std::accumulate(container.cbegin(), container.cend(), Result(0),
                                 [&](Result const& result, Item const& item) { return result + toNumber(item); });

    return sum / container.size();
};

template<typename Container, typename Conflict, typename Compare>
void resolveConflicts(Container& container, Conflict&& conflict, Compare&& comp) {
    if (container.size() < 2) return;

    for (auto itr = std::next(container.begin()); itr != container.end(); ) {
        auto itrPrev = std::prev(itr);
        if (conflict(*itrPrev, *itr)) {
            if (comp(*itrPrev, *itr)) {
                itr = std::next(container.erase(itrPrev));
            } else {
                itr = container.erase(itr);
            }
        } else {
            ++itr;
        }
    }
}

} // end namespace cuizhou

#endif //CUIZHOU_OCR_DATA_PROCESSING_H