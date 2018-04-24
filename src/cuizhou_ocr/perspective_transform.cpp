//
// Created by Zhihao Liu on 4/24/18.
//

#include <iostream>
#include "perspective_transform.h"


PerspectiveTransform::PerspectiveTransform(double scaleX, double scaleY, double offsetX, double offsetY)
        : scaleX_(scaleX), scaleY_(scaleY), offsetX_(offsetX), offsetY_(offsetY) {}

PerspectiveTransform::PerspectiveTransform(double scale, double offsetX, double offsetY)
        : PerspectiveTransform(scale, scale, offsetX, offsetY) {}

void PerspectiveTransform::setOffset(double offsetX, double offsetY) {
    offsetX_ = offsetX;
    offsetY_ = offsetY;
}

void PerspectiveTransform::setScale(double scaleX, double scaleY) {
    scaleX_ = scaleX;
    scaleY_ = scaleY;
}

void PerspectiveTransform::setScale(double uniformScale) {
    scaleX_ = scaleY_ = uniformScale;
}

void PerspectiveTransform::shift(double shiftX, double shiftY) {
    offsetX_ += shiftX;
    offsetY_ += shiftY;
}

void PerspectiveTransform::scale(double scaleX, double scaleY) {
    scaleX_ *= scaleX;
    scaleY_ *= scaleY;
}

void PerspectiveTransform::scale(double uniformScale) {
    scale(uniformScale, uniformScale);
}

PerspectiveTransform PerspectiveTransform::reversed() const {
    return PerspectiveTransform(1.0 / scaleX_,
                                1.0 / scaleY_,
                                -offsetX_ / scaleX_,
                                -offsetY_ / scaleY_);
}

cv::Point PerspectiveTransform::apply(cv::Point const& point) const {
    return cv::Point(int(std::round(point.x * scaleX_ + offsetX_)),
                     int(std::round(point.y * scaleY_ + offsetY_)));
}

cv::Rect PerspectiveTransform::apply(cv::Rect const& rect) const {
    return cv::Rect(int(std::round(rect.x * scaleX_ + offsetX_)),
                    int(std::round(rect.y * scaleY_ + offsetY_)),
                    int(std::round(rect.width * scaleX_)),
                    int(std::round(rect.height * scaleY_)));
}

PerspectiveTransform PerspectiveTransform::merge(PerspectiveTransform const& tr1, PerspectiveTransform const& tr2) {
    return PerspectiveTransform(tr1.scaleX_ * tr2.scaleX_,
                                tr1.scaleY_ * tr2.scaleY_,
                                tr1.offsetX_ * tr2.scaleX_ + tr2.offsetX_,
                                tr1.offsetY_ * tr2.scaleY_ + tr2.offsetY_);
}
PerspectiveTransform PerspectiveTransform::mergedWith(PerspectiveTransform const& that) {
    return merge(*this, that);
}

std::ostream& operator<<(std::ostream& strm, PerspectiveTransform const& obj) {
    return strm << "scale: "<< obj.scaleX_ << ", " << obj.scaleY_ << "; " << "offset: " << obj.offsetX_ << ", " << obj.offsetY_;
}
