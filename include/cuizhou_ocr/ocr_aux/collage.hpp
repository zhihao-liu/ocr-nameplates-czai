//
// Created by Zhihao Liu on 4/20/18.
//

#ifndef CUIZHOU_OCR_COLLAGE_H
#define CUIZHOU_OCR_COLLAGE_H

#include <opencv2/core/core.hpp>
#include "data_utils/enum_hashmap.hpp"
#include "data_utils/perspective_transform.h"

namespace cz {

template<typename FieldEnum>
class Collage {
private:
    struct RoiTransformInfo {
        cv::Rect roi;
        PerspectiveTransform backwardTransform;

        ~RoiTransformInfo();
        RoiTransformInfo();
        RoiTransformInfo(cv::Rect const& _roi, PerspectiveTransform const& _backwardTransform);
    };

public:
    ~Collage();
    Collage();

    Collage(cv::Mat const& image,
            EnumHashMap<FieldEnum, std::pair<cv::Rect, cv::Rect>> const& roiMapping,
            cv::Size const& resultSize);

    cv::Mat const& image() const;

    EnumHashMap<FieldEnum, std::vector<Detection>>
    splitDetections(std::vector<Detection> const& dets, float overlapThresh = 0.5);

private:
    cv::Mat resultImage_;
    EnumHashMap<FieldEnum, RoiTransformInfo> roiTransformInfos;
};

} // end namespace cz


#include "./impl/collage.impl.hpp"

#endif //CUIZHOU_OCR_COLLAGE_H
