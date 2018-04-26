#ifndef CUIZHOU_OCR_CLASSIFICATION_H
#define CUIZHOU_OCR_CLASSIFICATION_H

#include <string>
#include <opencv2/core/core.hpp>


namespace cuizhou {

struct Classification {
    std::string label;
    float score;

    ~Classification() = default;
    Classification() = default;

    Classification(std::string _label, float _score)
            : label(std::move(_label)), score(_score) {};
};

} // end namespace cuizhou

#endif //CUIZHOU_OCR_CLASSIFICATION_H
