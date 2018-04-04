//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "OcrNameplatesAlfa.h"
#include "PvaDetector.h"
#include "OcrUtils.hpp"


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cuizhou;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo/";
//    string pathInputDir = "/home/cuizhou/lzh/data/selected-test/";

    string pathModelKeys = "/home/cuizhou/lzh/models/table_header_model/car_brand_iter_100000.caffemodel";
    string pathPtKeys = "/home/cuizhou/lzh/models/table_header_model/test.prototxt";
    vector<string> classesKeys = OcrUtils::readClassNames("/home/cuizhou/lzh/models/table_header_model/classes_name.txt");

    string pathModelValues = "/home/cuizhou/lzh/models/single_char_model/alfa_engnum_char_iter_100000.caffemodel";
    string pathPtValues = "/home/cuizhou/lzh/models/single_char_model/test.prototxt";
    vector<string> classesChars = OcrUtils::readClassNames("/home/cuizhou/lzh/models/single_char_model/classes_name.txt");

    PVADetector detectorKeys;
    detectorKeys.init(pathPtKeys, pathModelKeys, classesKeys);
    detectorKeys.setComputeMode("gpu", 0);

    PVADetector detectorValues;
    detectorValues.init(pathPtValues, pathModelValues, classesChars);
    detectorValues.setComputeMode("gpu", 0);

    int countAll = 0, countCorrect = 0, countShorter = 0, countLonger = 0, countWrong = 0;

    for (directory_iterator itr(pathInputDir); itr != directory_iterator(); ++itr) {
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string id = fileName.substr(0, fileName.length() - 4);

        Mat img = imread(pathImg);
        if (img.empty()) continue;

        OcrNameplatesAlfa ocr(&detectorKeys, &detectorValues);
        ocr.setImage(img);
        ocr.processImage();

        InfoTable result = ocr.getResult();
        string value = result.get("Vin").value.content;

        ++countAll;
        if (value == id) {
            ++countCorrect;
        } else if (value.length() == id.length()) {
            ++countWrong;
        } else if (value.length() < id.length()) {
            ++countShorter;
        } else {
            ++countLonger;
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