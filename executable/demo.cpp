//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "detector.h"
#include "classifier.h"
#include "ocr_interface.h"
#include "ocr_aux/detection_proc.h"


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cz;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo";

    string dirPvaKeys = "/home/cuizhou/lzh/models/models_alfaromeo/pva_keys_compressed/";
    string ptPvaKeys = dirPvaKeys + "test.prototxt";
    string modelPvaKeys = dirPvaKeys + "car_brand_iter_100000.caffemodel";
    vector<string> classesPvaKeys = readClassNames(dirPvaKeys + "classes_name.txt", true);

    string dirPvaValueVin = "/home/cuizhou/lzh/models/models_alfaromeo/pva_vin_value_chars/";
    string ptPvaValueVin = dirPvaValueVin + "test.prototxt";
    string modelPvaValueVin = dirPvaValueVin + "alfa_engnum_char_iter_100000.caffemodel";
    vector<string> classesPvaValueVin = readClassNames(dirPvaValueVin + "classes_name.txt", true);

    string dirPvaValueOther = "/home/cuizhou/lzh/models/models_alfaromeo/pva_stitch_model/";
    string ptPvaValueOther = dirPvaValueOther + "merge_svd.prototxt";
    string modelPvaValueOther = dirPvaValueOther + "stitch_name_plate_iter_100000_merge_svd.caffemodel";
    vector<string> classesPvaValueOther = readClassNames(dirPvaValueOther + "classes_name.txt", true);

    string dirClassifierChars = "/home/cuizhou/lzh/models/models_alfaromeo/googlenet_chars/";
    string ptClassifierChars = dirClassifierChars + "deploy.prototxt";
    string modelClassifierChars = dirClassifierChars + "model_googlenet_iter_38942.caffemodel";
    string meanClassifierChars = dirClassifierChars + "mean.binaryproto";
    vector<string> classesClassifierChars = readClassNames(dirClassifierChars + "classname.txt");

    Detector detectorKeys;
    detectorKeys.init(ptPvaKeys, modelPvaKeys, classesPvaKeys);
    detectorKeys.setComputeMode("gpu", 0);

    Detector detectorValueVin;
    detectorValueVin.init(ptPvaValueVin, modelPvaValueVin, classesPvaValueVin);
    detectorValueVin.setComputeMode("gpu", 0);

    Detector detectorValueOther;
    detectorValueOther.init(ptPvaValueOther, modelPvaValueOther, classesPvaValueOther);
    detectorValueOther.setComputeMode("gpu", 0);

    Classifier classifierChars;
    classifierChars.init(ptClassifierChars, modelClassifierChars, meanClassifierChars, classesClassifierChars);

//    OcrInterface ocr(OcrType::NAMEPLATE_VOLKSWAGEN, {detectorValues});
    OcrInterface ocr(OcrType::NAMEPLATE_ALFAROMEO, {detectorKeys, detectorValueVin, detectorValueOther, classifierChars});

    auto start = std::chrono::system_clock::now();
    int countAll = 0, countCorrect = 0;

    for (directory_iterator itr(pathInputDir); itr != directory_iterator(); ++itr) {
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string imgId = fileName.substr(0, fileName.length() - 4);

        cout << "Processing " << imgId << "..." << endl;

        Mat img = imread(pathImg);
        if (img.empty()) continue;

        ocr.setImageSource(img);
        ocr.processImage();
        cout << ocr.getResultAsString() << endl;

        imshow("", ocr.drawResult());
//        imshow("", ocr.image());
        waitKey(0);
    }

    auto end = std::chrono::system_clock::now();

    cout << "Time elapsed: " << (end - start).count() << endl;

    return 0;
}