//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "classifier.h"
#include "pvadetector.h"
#include "ocr_nameplates_alfa.h"
#include "ocr_utils.hpp"

#define SAVE_RESULTS 1


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cuizhou;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo";
    string pathOutputDir = "/home/cuizhou/lzh/data/results";

    string modelPvaKeys = "/home/cuizhou/lzh/models/pva_keys_compressed/car_brand_iter_100000.caffemodel";
    string ptPvaKeys = "/home/cuizhou/lzh/models/pva_keys_compressed/test.prototxt";
    vector<string> classesPvaKeys = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_keys_compressed/classes_name.txt");

    string modelPvaValues1 = "/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/alfa_engnum_char_iter_100000.caffemodel";
    string ptPvaValues1 = "/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/test.prototxt";
    vector<string> classesPvaValues1 = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_vin_value_chars_compressed/classes_name.txt");

    string modelPvaValues2 = "/home/cuizhou/lzh/models/pva_other_value_chars_compressed/alfa_char_shape_pva_iter_100000.caffemodel";
    string ptPvaValues2 = "/home/cuizhou/lzh/models/pva_other_value_chars_compressed/test.prototxt";
    vector<string> classesPvaValues2 = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_other_value_chars_compressed/classes_name.txt");

    string modelGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/model_googlenet_iter_38942.caffemodel";
    string ptGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/deploy.prototxt";
    string meanGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/mean.binaryproto";
    string classesGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/classname.txt";

    PvaDetector detectorKeys;
    detectorKeys.init(ptPvaKeys, modelPvaKeys, classesPvaKeys);
    detectorKeys.setComputeMode("gpu", 0);

    PvaDetector detectorValues1;
    detectorValues1.init(ptPvaValues1, modelPvaValues1, classesPvaValues1);
    detectorValues1.setComputeMode("gpu", 0);

    PvaDetector detectorValues2;
    detectorValues2.init(ptPvaValues2, modelPvaValues2, classesPvaValues2);
    detectorValues2.setComputeMode("gpu", 0);

    Classifier classifier(ptGooglenet, modelGooglenet, meanGooglenet, classesGooglenet);

    auto start = std::chrono::system_clock::now();
    int countAll = 0, countCorrect = 0;

    for (directory_iterator itr(pathInputDir); itr != directory_iterator(); ++itr) {
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string imgId = fileName.substr(0, fileName.length() - 4);

        cout << "Processing " << imgId << "..." << endl;

        Mat img = imread(pathImg);
        if (img.empty()) continue;

        OcrNameplatesAlfa ocr(detectorKeys, detectorValues1, detectorValues2, classifier);
        ocr.setImage(img);
        ocr.processImage();
        auto result = ocr.getResultAsArray();
        for (auto const& item : result) {
            cout << item << endl;
        }
//        ocr.printResultToConsoleInChinese();
        imshow("", ocr.image());
        waitKey(0);

    }

    auto end = std::chrono::system_clock::now();

    cout << "Time elapsed: " << (end - start).count() << endl;

    return 0;
}