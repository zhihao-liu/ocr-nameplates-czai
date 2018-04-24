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
    struct TransformedRoiInfo {
        cv::Rect roi;
        PerspectiveTransform forwardTransform;

        ~TransformedRoiInfo() = default;
        TransformedRoiInfo() = default;
        TransformedRoiInfo(cv::Rect const& _roi, PerspectiveTransform const& _forwardTransform)
                : roi(_roi), forwardTransform(_forwardTransform) {};
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
    EnumHashMap<FieldEnum, TransformedRoiInfo> targetRoisInfo;
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

        TransformedRoiInfo roiInfo(targetRoi,
                                   imgToSubimg.mergedWith(subimgToTarget).mergedWith(targetToResultImg));

        targetRoisInfo.emplace(field, roiInfo);
    }

    resultImage_ = resultImg;
}

template<typename FieldEnum>
EnumHashMap<FieldEnum, std::vector<Detection>>
Collage<FieldEnum>::splitDetections(std::vector<Detection> const& dets, float overlapThresh) {
    EnumHashMap<FieldEnum, std::vector<Detection>> splitResult;
    for (auto const& roiInfo : targetRoisInfo) {
        PerspectiveTransform backwardTransform = roiInfo.second.forwardTransform.reversed();
        for (auto const& det : dets) {
            if (OcrUtils::computeAreaIntersection(det.getRect(), roiInfo.second.roi) > overlapThresh * det.getRect().area()) {
                Detection detCopy = det;
                detCopy.setRect(backwardTransform.apply(det.getRect()));
                splitResult[roiInfo.first].push_back(detCopy);
            }
        }
    }
    return splitResult;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_COLLAGE_H
