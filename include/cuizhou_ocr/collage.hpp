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
            std::vector<FieldEnum> const& fields,
            std::vector<cv::Rect> const& originRois,
            std::vector<cv::Rect> const& targetRois,
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
                            std::vector<FieldEnum> const& fields,
                            std::vector<cv::Rect> const& originRois,
                            std::vector<cv::Rect> const& targetRois,
                            cv::Size const& resultSize) {
    assert(fields.size() == originRois.size() && fields.size() == targetRois.size());

    cv::Mat resultImg(resultSize, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < fields.size(); ++i) {
        PerspectiveTransform imgToSubimg(1, -originRois[i].x, -originRois[i].y);
        PerspectiveTransform subimgToTarget;
        PerspectiveTransform targetToResultImg(1, targetRois[i].x, targetRois[i].y);

        cv::Mat resized = OcrUtils::imgResizeAndFill(image(originRois[i]), targetRois[i].size(), &subimgToTarget);
        resized.copyTo(resultImg(targetRois[i]));

        TransformedRoiInfo roiInfo(targetRois[i],
                                   imgToSubimg.mergedWith(subimgToTarget).mergedWith(targetToResultImg));
        targetRoisInfo.emplace(fields[i], roiInfo);

        // DEUBG
        {
            using namespace std;
            cout << "~~~~~~" << endl;
            cout << "TR1 -- " << imgToSubimg << endl;
            cout << "TR2 -- " << subimgToTarget << endl;
            cout << "TR3 -- " << targetToResultImg << endl;
            auto merged = imgToSubimg.mergedWith(subimgToTarget).mergedWith(targetToResultImg);
            cout << "MRG -- " << merged << endl;
            cout << "BKD -- " << merged.reversed() << endl;
            cout << "~~~~~~" << endl;
        }
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
                //DEBUG
                {
                    using namespace std;
                    cout << "TBKD -- " << backwardTransform << endl;
                    cout << "RECT -- " << det.getRect() << endl;
                }
                detCopy.setRect(backwardTransform.apply(det.getRect()));
                splitResult[roiInfo.first].push_back(detCopy);
            }
        }
    }
    return splitResult;
}

} // end namespace cuizhou


#endif //CUIZHOU_OCR_COLLAGE_H
