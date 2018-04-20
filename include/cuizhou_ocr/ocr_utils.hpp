//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef CUIZHOU_OCR_OCRUTILS_H
#define CUIZHOU_OCR_OCRUTILS_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include "detector.h"


namespace cuizhou {

class OcrUtils {
public:
    static cv::Mat imgResizeAndFill(cv::Mat const& img, int newWidth, int newHeight);
    static cv::Mat imgResizeAndFill(cv::Mat const& img, cv::Size const& newSize);
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

    template<typename T, typename F> static double findMedian(std::vector<T> vec, F const& mapToNum);
    template<typename T, typename F> static double computeMean(std::vector<T> const& vec, F const& mapToNum);

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

template<typename T, typename F>
double OcrUtils::findMedian(std::vector<T> vec, F const& mapToNum) {
    assert(!vec.empty());

    std::nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end(),
                     [&](T const& item1, T const& item2){ return mapToNum(item1) < mapToNum(item2); });

    return mapToNum(vec.at(vec.size() / 2));
};

template<typename T, typename F>
double OcrUtils::computeMean(std::vector<T> const& vec, F const& mapToNum) {
    assert(!vec.empty());

    double avg = 0;
    for (auto item: vec) {
        avg += mapToNum(item);
    }
    return avg / vec.size();
};

} // end namespace cuizhou


#endif //CUIZHOU_OCR_OCRUTILS_H
