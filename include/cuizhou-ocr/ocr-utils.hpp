//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef CUIZHOU_OCR_OCRUTILS_H
#define CUIZHOU_OCR_OCRUTILS_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include "pvadetector.h"


namespace cuizhou {
    class OcrUtils {
    public:
        static void imResizeAndFill(cv::Mat& img, int newWidth, int newHeight);
        static void imrotate(cv::Mat& img, cv::Mat& newImg, double angleInDegree);
        static std::vector<std::string> readClassNames(std::string const& path);

        static int xMid(cv::Rect const& rect);
        static int yMid(cv::Rect const& rect);

        static int computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
        static int computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
        static int computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2);
        static double computeIou(cv::Rect const& rect1, cv::Rect const& rect2);
        static int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2);

        static cv::Rect& validateWindow(cv::Rect& window, int width, int height);
        static cv::Rect& validateWindow(cv::Rect& roi, cv::Mat const& img);
        static cv::Rect& validateWindow(cv::Rect& roi, cv::Rect const& extent);

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
}


using namespace cuizhou;

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


#endif //CUIZHOU_OCR_OCRUTILS_H
