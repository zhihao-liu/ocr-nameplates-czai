// Edited by Zhihao Liu, Apr. 2018

#include "detector.h"
#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


namespace cuizhou {

void Detector::init(std::string const& def, std::string const& net, std::vector<std::string> const& classes) {
    m_classes = classes;
    m_net = std::make_shared<caffe::Net<float>>(def, caffe::TEST);
    m_net->CopyTrainedLayersFrom(net);
}

void Detector::setComputeMode(std::string const& mode, int id) {
    using namespace caffe;

    if (mode == "gpu") {
        Caffe::SetDevice(id);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }
}

void Detector::setThresh(float conf_thresh, float nms_thresh) {
    m_confThresh = conf_thresh;
    m_nmsThresh = nms_thresh;
}

std::vector<Detection> Detector::detect(cv::Mat const& img) const {
    using namespace std;
    using namespace cv;

    vector<Detection> dets;
    int rpn_num;
    float *pred = nullptr;

    if (img.empty()) return dets;

    int im_size_min = min(img.rows, img.cols);
    int im_size_max = max(img.rows, img.cols);
    float im_scale = float(SCALES) / im_size_min;

    if (round(im_scale * im_size_max) > MAX_SIZE) im_scale = float(MAX_SIZE) / im_size_max;
    float im_scale_x = floor(img.cols * im_scale / SCALE_MULTIPLE_OF) * SCALE_MULTIPLE_OF / img.cols;

    float im_scale_y = floor(img.rows * im_scale / SCALE_MULTIPLE_OF) * SCALE_MULTIPLE_OF / img.rows;
    int height = int(img.rows * im_scale_y);
    int width = int(img.cols * im_scale_x);

    cv::Mat cv_resized;
    float im_info[6];
    float *boxes = nullptr;

    const float* bbox_delt;
    const float* rois;
    const float* pred_cls;

    cv::Mat cv_new(img.rows, img.cols, CV_32FC3, Scalar(0, 0, 0));
    for (int h = 0; h < img.rows; ++h) {
        for (int w = 0; w < img.cols; ++w) {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);
        }
    }

    cv::resize(cv_new, cv_resized, cv::Size(width, height));

    im_info[0] = cv_resized.rows;
    im_info[1] = cv_resized.cols;
    im_info[2] = im_scale_x;
    im_info[3] = im_scale_y;
    im_info[4] = im_scale_x;
    im_info[5] = im_scale_y;

    float *data_buf = new float[height*width * 3];
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            data_buf[(0 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }

    int channel_nums = cv_resized.channels();
    m_net->blob_by_name("data")->Reshape(1, channel_nums, height, width);
    m_net->Reshape();

    float* input_data = m_net->blob_by_name("data")->mutable_cpu_data();
    vector<cv::Mat> input_channels;
    for (int i = 0; i < channel_nums; ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += height*width;
    }
    cv::split(cv_resized, input_channels);
    m_net->blob_by_name("im_info")->set_cpu_data(im_info);

    m_net->ForwardFrom(0);

    bbox_delt = m_net->blob_by_name("bbox_pred")->cpu_data();

    rpn_num = m_net->blob_by_name("rois")->num();
    rois = m_net->blob_by_name("rois")->cpu_data();
    pred_cls = m_net->blob_by_name("cls_prob")->cpu_data();
    boxes = new float[rpn_num * 4];

    for (int n = 0; n < rpn_num; n++) {
        for (int c = 0; c < 4; c++) {
            boxes[n * 4 + c] = rois[n * 5 + c + 1] / im_info[c + 2];
        }
    }
    pred = new float[rpn_num * 5 * m_classes.size()];
    bbox_transform_inv(rpn_num, bbox_delt, pred_cls, boxes, pred, img.rows, img.cols);

    float *pred_per_class = nullptr;
    float *sorted_pred_cls = nullptr;
    int *keep = nullptr;
    int num_out;

    pred_per_class = new float[rpn_num * 5];
    sorted_pred_cls = new float[rpn_num * 5];
    keep = new int[rpn_num];
    for (int i = 1; i < m_classes.size(); ++i)	{
        for (int j = 0; j < rpn_num; ++j) {
            for (int k = 0; k<5; k++) {
                pred_per_class[j * 5 + k] = pred[(i * rpn_num + j) * 5 + k];
            }
        }
        boxes_sort(rpn_num, pred_per_class, sorted_pred_cls);
        nms(keep, &num_out, sorted_pred_cls, rpn_num, 5, m_nmsThresh);
        vector<Detection> singleDets = overThresh(keep, num_out, sorted_pred_cls, m_confThresh, m_classes[i]);
        dets.insert(dets.end(), singleDets.begin(), singleDets.end());
    }

    delete[] data_buf; data_buf = nullptr;
    delete[] boxes; boxes = nullptr;
    delete keep; keep = nullptr;
    delete pred_per_class; pred_per_class = nullptr;
    delete sorted_pred_cls; sorted_pred_cls = nullptr;
    delete pred; pred = nullptr;

    return dets;
}

std::vector<Detection> Detector::detect(cv::Mat const& img, std::string const& class_mask) const {
    std::vector<Detection> dets = detect(img);
    std::vector<Detection> mask_dets;

    std::copy_if(dets.cbegin(), dets.cend(), std::back_inserter(mask_dets),
                 [&](Detection const& det) { return det.label == class_mask; });

    return mask_dets;
}

void Detector::drawBox(cv::Mat& img, std::vector<Detection> const& dets) {
    for (auto const& det : dets) {
        rectangle(img, det.rect, cv::Scalar(255, 0, 0), 1);
        std::ostringstream os;
        os.precision(2);
        os << det.score;

        putText(img,
                det.label,
                cv::Point(det.rect.x, det.rect.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 1);
    }
}

float Detector::iou(const float A[], const float B[]) {
    using namespace std;

    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) return 0;

    const float x1 = max(A[0], B[0]);
    const float y1 = max(A[1], B[1]);
    const float x2 = min(A[2], B[2]);
    const float y2 = min(A[3], B[3]);

    const float width = max(0.0f, x2 - x1 + 1.0f);
    const float height = max(0.0f, y2 - y1 + 1.0f);
    const float area = width * height;

    const float A_area = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
    const float B_area = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

    return area / (A_area + B_area - area);
}

void Detector::nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh) {
    using namespace std;

    int count = 0;
    vector<char> is_dead(boxes_num, 0);

    for (int i = 0; i < boxes_num; ++i) {
        if (is_dead[i]) continue;
        keep_out[count++] = i;
        for (int j = i + 1; j < boxes_num; ++j) {
            if (!is_dead[j] && iou(&boxes_host[i * 5], &boxes_host[j * 5])>nms_overlap_thresh) {
                is_dead[j] = 1;
            }
        }
    }
    *num_out = count;
    is_dead.clear();
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  boxes_sort
*  Description:  Sort the bounding box according score
* =====================================================================================
*/
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred) {
    using namespace std;

    vector<Info> my;
    Info tmp;
    for (int i = 0; i < num; i++) {
        tmp.score = pred[i * 5 + 4];
        tmp.head = pred + i * 5;
        my.push_back(tmp);
    }

    std::sort(my.begin(), my.end(),
              [](Info const& info1, Info const& info2) { return info1.score > info2.score; });

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < 5; j++) {
            sorted_pred[i * 5 + j] = my[i].head[j];
        }
    }
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  bbox_transform_inv
*  Description:  Compute bounding box regression value
* =====================================================================================
*/
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width) const {
    using namespace std;

    float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
    for (int i = 0; i < num; i++) {
        width = float(boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0);
        height = float(boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0);
        ctr_x = float(boxes[i * 4 + 0] + 0.5 * width);
        ctr_y = float(boxes[i * 4 + 1] + 0.5 * height);
        int class_num = int(m_classes.size());
        for (int j = 0; j < class_num; j++) {
            dx = box_deltas[(i*class_num + j) * 4 + 0];
            dy = box_deltas[(i*class_num + j) * 4 + 1];
            dw = box_deltas[(i*class_num + j) * 4 + 2];
            dh = box_deltas[(i*class_num + j) * 4 + 3];
            pred_ctr_x = ctr_x + width*dx;
            pred_ctr_y = ctr_y + height*dy;
            pred_w = width * exp(dw);
            pred_h = height * exp(dh);
            pred[(j*num + i) * 5 + 0] = float(max(min(pred_ctr_x - 0.5* pred_w, (img_width - 1)*1.0), 0.0));
            pred[(j*num + i) * 5 + 1] = float(max(min(pred_ctr_y - 0.5* pred_h, (img_height - 1)*1.0), 0.0));
            pred[(j*num + i) * 5 + 2] = float(max(min(pred_ctr_x + 0.5* pred_w, (img_width - 1)*1.0), 0.0));
            pred[(j*num + i) * 5 + 3] = float(max(min(pred_ctr_y + 0.5* pred_h, (img_height - 1)*1.0), 0.0));
            pred[(j*num + i) * 5 + 4] = pred_cls[i*class_num + j];
        }
    }
}

std::vector<Detection> Detector::overThresh(int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH, std::string label) const {
    using namespace std;
    using namespace cv;

    vector<Detection> dets;
    int i = 0;
    while (i < num_out) {
        if (sorted_pred_cls[keep[i] * 5 + 4] < CONF_THRESH)
            break;
        Detection det;
        det.label = label;
        det.rect = Rect(Point(int(round(sorted_pred_cls[keep[i] * 5 + 0])), int(round(sorted_pred_cls[keep[i] * 5 + 1]))),
                        Point(int(round(sorted_pred_cls[keep[i] * 5 + 2])), int(round(sorted_pred_cls[keep[i] * 5 + 3]))));
        det.score = sorted_pred_cls[keep[i] * 5 + 4];
        dets.push_back(det);
        ++i;
    }
    return dets;
}

} // end namespace cuizhou