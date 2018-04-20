//
// Created by Zhihao Liu on 4/20/18.
//

#ifndef CUIZHOU_OCR_COLLAGE_H
#define CUIZHOU_OCR_COLLAGE_H

#include <functional>
#include "ocr_utils.hpp"
#include "enum_hashmap.hpp"


namespace cuizhou {

template<typename FieldType>
class Collage {
public:
    ~Collage() = default;
    Collage() = default;

    Collage(std::vector<FieldType> const& fieldArray,
            std::vector<cv::Mat> const& imageArray,
            std::vector<cv::Rect> const& roiArray,
            cv::Size const& resultSize);

    cv::Mat const& image() const { return result_; };

private:
    cv::Mat result_;
    EnumHashMap<FieldType, cv::Rect> roiMap_;
};

template<typename FieldType>
Collage<FieldType>::Collage(std::vector<FieldType> const& fieldArray,
                            std::vector<cv::Mat> const& imageArray,
                            std::vector<cv::Rect> const& roiArray,
                            cv::Size const& resultSize) {
    assert(fieldArray.size() == imageArray.size() && fieldArray.size() == roiArray.size());

    cv::Mat result(resultSize, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < fieldArray.size(); ++i) {
        roiMap_.emplace(fieldArray[i], roiArray[i]);

        cv::Mat resized = OcrUtils::imgResizeAndFill(imageArray[i], roiArray[i].size());
        cv::Mat subResult = result(roiArray[i]);
        resized.copyTo(subResult);
    }

    result_ = result;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_COLLAGE_H
