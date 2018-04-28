//
// Created by Zhihao Liu on 4/24/18.
//

#ifndef CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H
#define CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H

#include <opencv2/core/core.hpp>


namespace cz {

// preserves parameters for a perspective transform for coordinates
// the transform is performed as x = x * scaleX + offsetX
class PerspectiveTransform {
public:
    ~PerspectiveTransform();
    PerspectiveTransform();

    PerspectiveTransform(double scaleX, double scaleY, double offsetX, double offsetY);
    PerspectiveTransform(double scale, double offsetX, double offsetY);

    void setOffset(double offsetX, double offsetY);
    void setScale(double uniformScale);
    void setScale(double scaleX, double scaleY);

    void shiftBy(double shiftX, double shiftY);
    void scaleBy(double uniformScale);
    void scaleBy(double scaleX, double scaleY);

    cv::Point apply(cv::Point const& point) const;
    cv::Rect apply(cv::Rect const& rect) const;

    PerspectiveTransform& reverse();
    PerspectiveTransform reversed() const;

    PerspectiveTransform& merge(PerspectiveTransform const& that);
    PerspectiveTransform merged(PerspectiveTransform const& that) const;

    friend std::ostream& operator<<(std::ostream& strm, PerspectiveTransform const& obj);

private:
    double offsetX_ = 0, offsetY_ = 0;
    double scaleX_ = 1, scaleY_ = 1;
};

}


#endif //CUIZHOU_OCR_PERSPECTIVE_TRANSFORM_H
