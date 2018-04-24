//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef CUIZHOU_OCR_OCRUTILS_H
#define CUIZHOU_OCR_OCRUTILS_H

#include <fstream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include "detector.h"
#include "perspective_transform.h"


namespace cuizhou {

class OcrUtils {
public:
    static cv::Mat imgResizeAndFill(cv::Mat const& img,
                                    int newWidth, int newHeight,
                                    PerspectiveTransform* pForwardTransform = nullptr);
    static cv::Mat imgResizeAndFill(cv::Mat const& img,
                                    cv::Size const& newSize,
                                    PerspectiveTransform* pForwardTransform = nullptr);

    static cv::Mat imgRotate(cv::Mat const& img, double angleInDegree);

    static std::vector<std::string> readClassNames(std::string const& path);

    static int xMid(cv::Rect const& rect);
    static int yMid(cv::Rect const& rect);

    static int computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
    static int computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
    static int computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2);
    static float computeIou(cv::Rect const& rect1, cv::Rect const& rect2);
    static int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2);

    static cv::Rect validateRoi(cv::Rect const& roi, int width, int height);
    static cv::Rect validateRoi(cv::Rect const& roi, cv::Mat const& img);
    static cv::Rect validateRoi(cv::Rect const& roi, cv::Rect const& extent);

    template<typename Item>
    static Item findMedian(std::vector<Item> const& vec);
    template<typename Item, typename Func>
    static auto findMedian(std::vector<Item> const& vec, Func&& toNumber) -> typename std::result_of<Func(Item)>::type;

    template<typename Item>
    static Item computeMean(std::vector<Item> const& vec);
    template<typename Item, typename Func>
    static auto computeMean(std::vector<Item> const& vec, Func&& toNumber) -> typename std::result_of<Func(Item)>::type;

    static bool isNumbericChar(std::string const& str);
};


class LeastSquare {
public:
    LeastSquare(std::vector<double> const& x, std::vector<double> const& y);
    double getSlope() const;
    double getConstant() const;
private:
    double a, b;
};

// find the median of a collection of numbers
template<typename Item>
Item OcrUtils::findMedian(std::vector<Item> const& vec) {
    assert(!vec.empty());

    std::vector<Item> vecCopy = vec;
    std::nth_element(vecCopy.begin(), vecCopy.begin() + vecCopy.size() / 2, vecCopy.end());

    return vecCopy.at(vecCopy.size() / 2);
};

// find the median of a collection with a given functor that maps each item to a number
template<typename Item, typename Func>
auto OcrUtils::findMedian(std::vector<Item> const& vec, Func&& toNumber) -> typename std::result_of<Func(Item)>::type {
    assert(!vec.empty());

    using Result = typename std::result_of<Func(Item)>::type;

    std::vector<Result> numbers;
    std::transform(vec.cbegin(), vec.cend(), std::back_inserter(numbers), [&](Item const& item) { return toNumber(item); });
    std::nth_element(numbers.begin(), numbers.begin() + numbers.size() / 2, numbers.end());

    return numbers.at(numbers.size() / 2);
};

// compute the mean of a collection of numbers
template<typename Item>
Item OcrUtils::computeMean(std::vector<Item> const& vec) {
    assert(!vec.empty());

    Item sum = std::accumulate(vec.cbegin(), vec.cend(), Item(0));

    return sum / vec.size();
};

// compute the mean of a collection with a given functor that maps each item to a number
template<typename Item, typename Func>
auto OcrUtils::computeMean(std::vector<Item> const& vec, Func&& toNumber) -> typename std::result_of<Func(Item)>::type {
    assert(!vec.empty());

    using Result = typename std::result_of<Func(Item)>::type;

    Result sum = std::accumulate(vec.cbegin(), vec.cend(), Result(0),
                                 [&](Result const& result, Item const& item) { return result + toNumber(item); });

    return sum / vec.size();
};

} // end namespace cuizhou


#endif //CUIZHOU_OCR_OCRUTILS_H
