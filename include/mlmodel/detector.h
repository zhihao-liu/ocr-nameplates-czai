#ifndef DETECTOR_H
#define DETECTOR_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "mlmodel.h"
#include "detection.h"


namespace cuizhou {

class Detector : public MlModel {
public:
	~Detector() = default;
	Detector() = default;

	void init(std::string const& net_pt,
			  std::string const& net_weights,
			  std::vector<std::string> const& classes);

	void setComputeMode(std::string const& mode = "cpu", int id = 0);
	void setThresh(float conf_thresh = 0.7, float nms_thresh = 0.3);

	std::vector<Detection> detect(cv::Mat const& img) const;
	std::vector<Detection> detect(cv::Mat const& img, std::string const& class_mask) const;

	static void drawBox(cv::Mat& img, std::vector<Detection> const& dets);

private:
	std::vector<std::string> m_classes;
	std::shared_ptr<caffe::Net<float>> m_net;
	float m_confThresh;
	float m_nmsThresh;

	static int const SCALE_MULTIPLE_OF = 32;
	static int const MAX_SIZE = 1280;
	static int const SCALES = 640;

	void detectAux(cv::Mat img, float* pred, int& rpn_num) const;
	std::vector<Detection> overThresh(int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH, std::string className) const;

	static float iou(const float A[], const float B[]);
	static void nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh);
	static void boxes_sort(int num, const float* pred, float* sorted_pred);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width) const;

	struct Info {
		float score;
		const float* head;
	};
};

}

#endif //DETECTOR_H
