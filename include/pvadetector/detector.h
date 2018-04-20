#ifndef DETECTOR_H
#define DETECTOR_H
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include "detection.h"
#include <vector>

class Detector
{
public:
	Detector() {}
	void init(std::string net_pt, std::string net_weights, std::vector<std::string> classes);
	void setComputeMode(std::string mode = "cpu", int id = 0);
	void setThresh(float conf_thresh = 0.7, float nms_thresh = 0.3);
    std::vector<Detection> detect(cv::Mat const& img) const;
    std::vector<Detection> detect(cv::Mat const& img, std::string class_mask) const;
	void detect_aux(cv::Mat img, float* pred, int& rpn_num) const;
	static void drawBox(cv::Mat img, std::vector<Detection> dets);
	std::vector<Detection> overThresh(int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH, std::string className) const;

private:
	static float iou(const float A[], const float B[]);
	static void nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
		int boxes_dim, float nms_overlap_thresh);
	void bbox_transform_inv(const int num, const float* box_deltas,
		const float* pred_cls, float* boxes, float* pred, int img_height, int img_width) const;
	static void boxes_sort(int num, const float* pred, float* sorted_pred);

	std::vector<std::string> m_classes;
	std::shared_ptr<caffe::Net<float> > m_net;
	float m_confThresh;
	float m_nmsThresh;

	const int SCALE_MULTIPLE_OF = 32;
	const int MAX_SIZE = 1280; // 2000;
	const int SCALES = 640;
};

#endif //DETECTOR_H
