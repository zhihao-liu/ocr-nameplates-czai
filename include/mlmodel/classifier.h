#ifndef FULL_VEHICLE_INFOMATION_CLASSIFIER_H
#define FULL_VEHICLE_INFOMATION_CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include "mlmodel.h"
#include "classification.h"


namespace cuizhou {

class Classifier : public MlModel {
public:
    ~Classifier() = default;
    Classifier() = default;

    void init(std::string const& model_file,
              std::string const& trained_file,
              std::string const& mean_file,
              std::vector<std::string> const& label);

    std::vector<Classification> classify(cv::Mat const& img, int n = 5) const;

private:
    std::shared_ptr<caffe::Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<std::string> labels_;

    void setMean(std::string const& mean_file);
    std::vector<float> predict(cv::Mat const& img) const;
    void wrapInputLayer(std::vector<cv::Mat>& input_channels) const;
    void preprocess(cv::Mat const& img, std::vector<cv::Mat>& input_channels) const;
    static std::vector<int> argmax(std::vector<float> const& v, int n);
};

} // end namespace cuizhou

#endif //FULL_VEHICLE_INFOMATION_CLASSIFIER_H
