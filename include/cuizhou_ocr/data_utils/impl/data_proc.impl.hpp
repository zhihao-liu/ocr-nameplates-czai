namespace cz {

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
template<typename Container, typename ConflictPred, typename Compare>
void resolveConflicts(Container& container, ConflictPred&& conflict, Compare&& comp) {
    if (container.size() < 2) return;

    for (auto itr1 = container.begin(); itr1 != container.end(); ) {
        bool erasedItr1 = false;

        for (auto itr2 = container.begin(); itr2 != container.end(); ) {
            if (itr1 == itr2 || !conflict(*itr1, *itr2)) {
                ++itr2;
                continue;
            }

            if (comp(*itr1, *itr2)) {
                itr1 = container.erase(itr1);
                erasedItr1 = true;
                break;
            } else {
                itr2 = container.erase(itr2);
                continue;
            }
        }

        if (!erasedItr1) ++itr1;
    }

//    for (auto itr = std::next(container.begin()); itr != container.end(); ) {
//        auto itrPrev = std::prev(itr);
//        if (conflict(*itrPrev, *itr)) {
//            if (comp(*itrPrev, *itr)) {
//                itr = std::next(container.erase(itrPrev));
//            } else {
//                itr = container.erase(itr);
//            }
//        } else {
//            ++itr;
//        }
//    }
}

template<typename Container, typename Field, typename RefObj, typename RelevancePred>
auto distributeItemsByField(Container const& items, EnumHashMap<Field, RefObj> const& refMap, RelevancePred&& relevant)
-> EnumHashMap<Field, Container> {
    EnumHashMap<Field, Container> distributions;

    for (auto const& refItem : refMap) {
        for (auto const& item : items) {
            if (relevant(item, refItem.second)) {
                distributions[refItem.first].push_back(item);
            }
        }
    }

    return distributions;
}

}; // end namespace cz
