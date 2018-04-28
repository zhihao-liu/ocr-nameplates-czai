#include "data_utils/cv_extension.h"
#include "data_utils/data_proc.hpp"
#include "ocr_aux/detection_proc.h"

namespace cz {

template <typename FieldEnum>
Collage<FieldEnum>::RoiTransformInfo::~RoiTransformInfo() = default;

template <typename FieldEnum>
Collage<FieldEnum>::RoiTransformInfo::RoiTransformInfo() = default;

template <typename FieldEnum>
Collage<FieldEnum>::RoiTransformInfo::RoiTransformInfo(cv::Rect const& _roi, PerspectiveTransform const& _backwardTransform)
        : roi(_roi), backwardTransform(_backwardTransform) {};

template<typename FieldEnum>
Collage<FieldEnum>::~Collage() = default;

template<typename FieldEnum>
Collage<FieldEnum>::Collage() = default;

template<typename FieldEnum>
cv::Mat const& Collage<FieldEnum>::image() const {
    return resultImage_;
}

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

        cv::Mat resized = imgResizeAndFill(image(originRoi), targetRoi.size(), &subimgToTarget);
        resized.copyTo(resultImg(targetRoi));

        RoiTransformInfo roiInfo(targetRoi,
                                 imgToSubimg.merge(subimgToTarget).merge(targetToResultImg).reverse());

        roiTransformInfos.emplace(field, roiInfo);
    }

    resultImage_ = resultImg;
}

template<typename FieldEnum>
EnumHashMap<FieldEnum, std::vector<Detection>> Collage<FieldEnum>::splitDetections(std::vector<Detection> const& dets, float overlapThresh) {
    EnumHashMap<FieldEnum, std::vector<Detection>> splitResults =
            distributeItemsByField(dets, roiTransformInfos,
                                   [&](Detection const& det, RoiTransformInfo const& roiInfo) {
                                       return (det.rect & roiInfo.roi).area() > overlapThresh * det.rect.area();
                                   });

    for (auto& fieldResult : splitResults) {
        auto itrRoiInfo = roiTransformInfos.find(fieldResult.first);
        if (itrRoiInfo == roiTransformInfos.end()) continue;

        for (auto& det : fieldResult.second) {
            det.rect = itrRoiInfo->second.backwardTransform.apply(det.rect);
        }
    }

    return splitResults;
}

} // end namespace cz