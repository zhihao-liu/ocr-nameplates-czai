//
// Created by Zhihao Liu on 4/24/18.
//

#include <iostream>
#include "data_utils/perspective_transform.h"


namespace cz {

PerspectiveTransform::~PerspectiveTransform() = default;

PerspectiveTransform::PerspectiveTransform() = default;

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

void PerspectiveTransform::shiftBy(double shiftX, double shiftY) {
    offsetX_ += shiftX;
    offsetY_ += shiftY;
}

void PerspectiveTransform::scaleBy(double scaleX, double scaleY) {
    scaleX_ *= scaleX;
    scaleY_ *= scaleY;
}

void PerspectiveTransform::scaleBy(double uniformScale) {
    scaleBy(uniformScale, uniformScale);
}

PerspectiveTransform& PerspectiveTransform::reverse() {
    scaleX_ = 1.0 / scaleX_;
    scaleY_ = 1.0 / scaleY_;
    offsetX_ *= -scaleX_;
    offsetY_ *= -scaleY_;
    return *this;
}

PerspectiveTransform PerspectiveTransform::reversed() const {
    PerspectiveTransform copy = *this;
    return copy.reverse();
}

cv::Point PerspectiveTransform::apply(cv::Point const& point) const {
    return cv::Point(static_cast<int>(std::round(point.x * scaleX_ + offsetX_)),
                     static_cast<int>(std::round(point.y * scaleY_ + offsetY_)));
}

cv::Rect PerspectiveTransform::apply(cv::Rect const& rect) const {
    return cv::Rect(static_cast<int>(std::round(rect.x * scaleX_ + offsetX_)),
                    static_cast<int>(std::round(rect.y * scaleY_ + offsetY_)),
                    static_cast<int>(std::round(rect.width * scaleX_)),
                    static_cast<int>(std::round(rect.height * scaleY_)));
}

PerspectiveTransform& PerspectiveTransform::merge(PerspectiveTransform const& that) {
    scaleX_ *= that.scaleX_;
    scaleY_ *= that.scaleY_;
    offsetX_ = offsetX_ * that.scaleX_ + that.offsetX_;
    offsetY_ = offsetY_ * that.scaleY_ + that.offsetY_;
    return *this;
}

PerspectiveTransform PerspectiveTransform::merged(PerspectiveTransform const& that) const {
    PerspectiveTransform copy = *this;
    return copy.merge(that);
}

std::ostream& operator<<(std::ostream& strm, PerspectiveTransform const& obj) {
    return strm << "scale: {" << obj.scaleX_ << ", " << obj.scaleY_ << "}; "
                << "offset: {" << obj.offsetX_ << ", " << obj.offsetY_ << "}";
}

}
