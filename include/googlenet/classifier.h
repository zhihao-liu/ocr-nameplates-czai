//
// Created by wz on 17-11-8.
//

#ifndef FULL_VEHICLE_INFOMATION_CLASSIFIER_H
#define FULL_VEHICLE_INFOMATION_CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;
using std::string;
typedef std::pair<string, float> Prediction;
class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file);

    std::vector<Prediction> classify(const cv::Mat &img, int N = 5) const;

private:
    void setMean(const string &mean_file);

    std::vector<float> predict(const cv::Mat &img) const;

    void wrapInputLayer(std::vector<cv::Mat> *input_channels) const;

    void preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels) const;

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};

#endif //FULL_VEHICLE_INFOMATION_CLASSIFIER_H
