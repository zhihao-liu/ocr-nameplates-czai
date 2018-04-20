//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "classifier.h"
#include "detector.h"
#include "ocr_nameplates_alfa.h"
#include "ocr_utils.hpp"

#define SAVE_RESULTS 1


// DEBUG
extern cv::Mat d_imgToShow;

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cuizhou;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo";
    string pathOutputDir = "/home/cuizhou/lzh/data/results";

    string ptPvaKeys = "/home/cuizhou/lzh/models/pva_keys_compressed/test.prototxt";
    string modelPvaKeys = "/home/cuizhou/lzh/models/pva_keys_compressed/car_brand_iter_100000.caffemodel";
    vector<string> classesPvaKeys = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_keys_compressed/classes_name.txt");

    string ptPvaValuesVin = "/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/test.prototxt";
    string modelPvaValuesVin = "/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/alfa_engnum_char_iter_100000.caffemodel";
    vector<string> classesPvaValuesVin = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/classes_name.txt");

    string ptPvaValuesOthers = "/home/cuizhou/lzh/models/pva_other_value_chars_compressed/test.prototxt";
    string modelPvaValuesOthers = "/home/cuizhou/lzh/models/pva_other_value_chars_compressed/alfa_char_shape_pva_iter_100000.caffemodel";
    vector<string> classesPvaValuesOthers = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_other_value_chars_compressed/classes_name.txt");

    string ptPvaValuesStitched = "/home/cuizhou/lzh/models/pva_stitch_model/merge_svd.prototxt";
    string modelPvaValuesStitched = "/home/cuizhou/lzh/models/pva_stitch_model/stitch_name_plate_iter_100000_merge_svd.caffemodel";
    vector<string> classesPvaValuesStitched = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_stitch_model/classes_name.txt");

    string ptGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/deploy.prototxt";
    string modelGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/model_googlenet_iter_38942.caffemodel";
    string meanGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/mean.binaryproto";
    string classesGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/classname.txt";

    Detector detectorKeys;
    detectorKeys.init(ptPvaKeys, modelPvaKeys, classesPvaKeys);
    detectorKeys.setComputeMode("gpu", 0);

    Detector detectorValuesVin;
    detectorValuesVin.init(ptPvaValuesVin, modelPvaValuesVin, classesPvaValuesVin);
    detectorValuesVin.setComputeMode("gpu", 0);

    Detector detectorValuesOthers;
    detectorValuesOthers.init(ptPvaValuesOthers, modelPvaValuesOthers, classesPvaValuesOthers);
    detectorValuesOthers.setComputeMode("gpu", 0);
    
    Detector detectorValuesStitched;
    detectorValuesStitched.init(ptPvaValuesStitched, modelPvaValuesStitched, classesPvaValuesStitched);
    detectorValuesStitched.setComputeMode("gpu", 0);

    Classifier classifier(ptGooglenet, modelGooglenet, meanGooglenet, classesGooglenet);

    OcrNameplatesAlfa ocr(detectorKeys, detectorValuesVin, detectorValuesOthers, detectorValuesStitched, classifier);

    auto start = std::chrono::system_clock::now();
    int countAll = 0, countCorrect = 0;

    for (directory_iterator itr(pathInputDir); itr != directory_iterator(); ++itr) {
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string imgId = fileName.substr(0, fileName.length() - 4);

        cout << "Processing " << imgId << "..." << endl;

        Mat img = imread(pathImg);
        if (img.empty()) continue;

        ocr.setImage(img);
        ocr.processImage();
        auto result = ocr.getResultAsArray();
        for (auto const& item : result) {
            cout << item << endl;
        }

//        imshow("", ocr.image());
        imshow("", d_imgToShow);
        waitKey(0);
    }

    auto end = std::chrono::system_clock::now();

    cout << "Time elapsed: " << (end - start).count() << endl;

    return 0;
}