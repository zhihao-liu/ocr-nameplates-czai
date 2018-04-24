//
// Created by Zhihao Liu on 4/24/18.
//

#ifndef CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H
#define CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H

#include <opencv2/core/core.hpp>


// preserves parameters for a perspective transform for coordinates
// the transform is performed as x = x * scaleX + offsetX
class PerspectiveTransform {
public:
    ~PerspectiveTransform() = default;
    PerspectiveTransform() = default;
    PerspectiveTransform(double scaleX, double scaleY, double offsetX, double offsetY);
    PerspectiveTransform(double scale, double offsetX, double offsetY);

    PerspectiveTransform reversed() const;
    void setOffset(double offsetX, double offsetY);
    void setScale(double uniformScale);
    void setScale(double scaleX, double scaleY);
    void shift(double shiftX, double shiftY);
    void scale(double uniformScale);
    void scale(double scaleX, double scaleY);

    cv::Point apply(cv::Point const& point) const;
    cv::Rect apply(cv::Rect const& rect) const;

    static PerspectiveTransform merge(PerspectiveTransform const& tr1, PerspectiveTransform const& tr2);
    PerspectiveTransform mergedWith(PerspectiveTransform const& that);

    friend std::ostream& operator<<(std::ostream& strm, PerspectiveTransform const& obj);

private:
    double offsetX_ = 0, offsetY_ = 0;
    double scaleX_ = 1, scaleY_ = 1;
};

#endif //CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H
