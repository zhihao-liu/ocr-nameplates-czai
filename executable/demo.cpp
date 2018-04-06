//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "classifier.h"
#include "PvaDetector.h"
#include "OcrNameplatesAlfa.h"
#include "OcrUtils.hpp"

#define SAVE_RESULTS 1


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cuizhou;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo/";
    string pathOutputDir = "/home/cuizhou/lzh/data/results-with-text/";

    string modelPvaKeys = "/home/cuizhou/lzh/models/pva_keys/car_brand_iter_100000.caffemodel";
    string ptPvaKeys = "/home/cuizhou/lzh/models/pva_keys/test.prototxt";
    vector<string> classesPvaKeys = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_keys/classes_name.txt");

    string modelPvaValues = "/home/cuizhou/lzh/models/pva_vin_value_chars/alfa_engnum_char_iter_100000.caffemodel";
    string ptPvaValues = "/home/cuizhou/lzh/models/pva_vin_value_chars/test.prototxt";
    vector<string> classesPvaValues = OcrUtils::readClassNames("/home/cuizhou/lzh/models/pva_vin_value_chars/classes_name.txt");

    string modelGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/model_googlenet_iter_38942.caffemodel";
    string ptGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/deploy.prototxt";
    string meanGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/mean.binaryproto";
    string classesGooglenet = "/home/cuizhou/lzh/models/googlenet_chars/classname.txt";

    PvaDetector detectorKeys;
    detectorKeys.init(ptPvaKeys, modelPvaKeys, classesPvaKeys);
    detectorKeys.setComputeMode("gpu", 0);

    PvaDetector detectorValues;
    detectorValues.init(ptPvaValues, modelPvaValues, classesPvaValues);
    detectorValues.setComputeMode("gpu", 0);

    Classifier classifier(ptGooglenet, modelGooglenet, meanGooglenet, classesGooglenet);


    int countAll = 0, countCorrect = 0, countShorter = 0, countLonger = 0, countWrong = 0;

    for (directory_iterator itr(pathInputDir); itr != directory_iterator(); ++itr) {
        // string pathImg = "/home/cuizhou/lzh/data/raw-alfaromeo/ZAREAEBN4H7546964.jpg";
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string imgId = fileName.substr(0, fileName.length() - 4);

        Mat img = imread(pathImg);
        if (img.empty()) continue;

        OcrNameplatesAlfa ocr(detectorKeys, detectorValues, classifier);
        ocr.setImage(img);
        ocr.processImage();

        InfoTable result = ocr.getResult();
        auto resultVin = result.get("Vin");
        if (resultVin == nullptr) continue;
        string valueVin = resultVin->value.content;

       img = ocr.getImage().clone();

        ++countAll;
        if (valueVin == imgId) {
            ++countCorrect;

            if (SAVE_RESULTS) {
                putText(img, "VIN", Point(resultVin->key.rect.x + 10, resultVin->key.rect.br().y + 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
                putText(img, valueVin, Point(resultVin->value.rect.x + 10, resultVin->value.rect.br().y + 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
                imwrite(pathOutputDir + "correct/" + imgId + "-result.jpg", img);
            }
        } else {
            if (valueVin.length() == imgId.length()) {
                ++countWrong;
            } else if (valueVin.length() < imgId.length()) {
                ++countShorter;
            } else {
                ++countLonger;
            }

            if (SAVE_RESULTS) {
                putText(img, "VIN", Point(resultVin->key.rect.x + 10, resultVin->key.rect.br().y + 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
                putText(img, valueVin, Point(resultVin->value.rect.x + 10, resultVin->value.rect.br().y + 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 127, 255), 2);
                imwrite(pathOutputDir + "incorrect/" + imgId + "-result.jpg", img);
            }
        }

        cout << countCorrect << " out of " << countAll << " (" << 100 * float(countCorrect) / countAll << "%) correct."
             << "...";
        cout << countWrong << " out of " << countAll << " (" << 100 * float(countWrong) / countAll << "%) wrong."
             << "...";
        cout << countShorter << " out of " << countAll << " (" << 100 * float(countShorter) / countAll << "%) shorter."
             << "...";
        cout << countLonger << " out of " << countAll << " (" << 100 * float(countLonger) / countAll << "%) longer."
             << endl;
    }

    waitKey(0);
    return 0;
}