//
// Created by Zhihao Liu on 4/20/18.
//

#ifndef CUIZHOU_OCR_COLLAGE_H
#define CUIZHOU_OCR_COLLAGE_H

#include <functional>
#include "ocr_utils.hpp"
#include "enum_hashmap.hpp"
#include "perspective_transform.h"


namespace cuizhou {

template<typename FieldEnum>
class Collage {
private:
    struct RoiTransformInfo {
        cv::Rect roi;
        PerspectiveTransform backwardTransform;

        ~RoiTransformInfo() = default;
        RoiTransformInfo() = default;
        RoiTransformInfo(cv::Rect _roi, PerspectiveTransform _backwardTransform)
                : roi(std::move(_roi)), backwardTransform(std::move(_backwardTransform)) {};
    };

public:
    ~Collage() = default;
    Collage() = default;

    Collage(cv::Mat const& image,
            EnumHashMap<FieldEnum, std::pair<cv::Rect, cv::Rect>> const& roiMapping,
            cv::Size const& resultSize);

    cv::Mat const& image() const { return resultImage_; };

    EnumHashMap<FieldEnum, std::vector<Detection>>
    splitDetections(std::vector<Detection> const& dets, float overlapThresh = 0.5);

private:
    cv::Mat resultImage_;
    EnumHashMap<FieldEnum, RoiTransformInfo> roiTransformInfos;
};

template<typename FieldEnum>
Collage<FieldEnum>::Collage(cv::Mat const& image,
                            EnumHashMap<FieldEnum, std::pair<cv::Rect, cv::Rect>> const& roiMapping,
                            cv::Size const& resultSize) {
    cv::Mat resultImg(resultSize, CV_8UC3, cv::Scalar(0, 0, 0));

    for (auto const& item : roiMapping) {
        FieldEnum field = item.first;
        cv::Rect const& originRoi = item.second.first;
        cv::Rect const& targetRoi = item.second.second;

        PerspectiveTransform imgToSubimg(1, -originRoi.x, -originRoi.y);
        PerspectiveTransform subimgToTarget;
        PerspectiveTransform targetToResultImg(1, targetRoi.x, targetRoi.y);

        cv::Mat resized = OcrUtils::imgResizeAndFill(image(originRoi), targetRoi.size(), &subimgToTarget);
        resized.copyTo(resultImg(targetRoi));

        RoiTransformInfo roiInfo(targetRoi,
                                 imgToSubimg.merge(subimgToTarget).merge(targetToResultImg).reverse());

        roiTransformInfos.emplace(field, roiInfo);
    }

    resultImage_ = resultImg;
}

template<typename FieldEnum>
EnumHashMap<FieldEnum, std::vector<Detection>>
Collage<FieldEnum>::splitDetections(std::vector<Detection> const& dets, float overlapThresh) {
    EnumHashMap<FieldEnum, std::vector<Detection>> splitResult;
    for (auto const& roiInfo : roiTransformInfos) {
        for (auto const& det : dets) {
            if (OcrUtils::computeAreaIntersection(det.rect, roiInfo.second.roi) > overlapThresh * det.rect.area()) {
                Detection transformedDet = det;
                transformedDet.rect = roiInfo.second.backwardTransform.apply(det.rect);
                splitResult[roiInfo.first].push_back(std::move(transformedDet));
            }
        }
    }
    return splitResult;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_COLLAGE_H
