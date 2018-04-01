//
// Created by Zhihao Liu on 18-3-21.
//

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pvaDetector.h"

using namespace std;
using namespace cv;


vector<string> readClassNames(string const& path) {
    ifstream file(path);
    vector<string> classNames;
    classNames.push_back("__background__");

    string line;
    while (getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
};

string join(vector<Detection> const& dets) {
    string str = "";
    for (auto det: dets) {
        str += det.getClass();
    }
    return str;
}

void printAll(unordered_map<string, string> const& map) {
    for (auto pair: map) {
        cout << pair.first << ": " << pair.second << endl;
    }
};

int main(int argc, char* argv[]) {
    string name = "ZAREAEBN1H7548459";
    string pathImg = "/home/cuizhou/lzh/data/raw-alfaromeo/" + name + ".jpg";

    string pathModelKeys = "/home/cuizhou/lzh/models/table_header_model/car_brand_iter_100000.caffemodel";
    string pathPtKeys = "/home/cuizhou/lzh/models/table_header_model/test.prototxt";
    vector<string> classesKeys = readClassNames("/home/cuizhou/lzh/models/table_header_model/classes_name.txt");

    string pathModelValues = "/home/cuizhou/lzh/models/single_char_model/alfa_engnum_char_iter_100000.caffemodel";
    string pathPtValues = "/home/cuizhou/lzh/models/single_char_model/test.prototxt";
    vector<string> classesChars = readClassNames("/home/cuizhou/lzh/models/single_char_model/classes_name.txt");


    PVADetector detectorKeys;
    detectorKeys.init(pathPtKeys, pathModelKeys, classesKeys);
    detectorKeys.setThresh(0.5, 0.1);
    detectorKeys.setComputeMode("gpu", 0);

    PVADetector detectorValues;
    detectorValues.init(pathPtValues, pathModelValues, classesChars);
    detectorValues.setThresh(0.05, 0.3);
    detectorValues.setComputeMode("gpu", 0);

    Mat img = imread(pathImg);
    unordered_map<string, string> infoTable;

    vector<Detection> keyDets = detectorKeys.detect(img);
    for (auto keyDet: keyDets) {
        string key = keyDet.getClass();
        string value;

        if (key == "Manufacturer") {

        } else if (key == "Brand") {

        } else if (key == "MaxMassAllowed") {

        } else if (key == "MaxNetPowerOfEngine") {

        } else if (key == "Country") {

        } else if (key == "Factory") {

        } else if (key == "EngineModel") {

        } else if (key == "NumPassengers") {

        } else if (key == "VehicleId") {
            Rect keyRect = keyDet.getRect();
            Rect valueRect(keyRect.br().x + 10, keyRect.tl().y - (keyRect.height * 0.1), keyRect.width * 1.35, keyRect.height * 1.2);
            vector<Detection> valueDets = detectorValues.detect(img(valueRect));

            sort(valueDets, [](Detection det1, Detection det2){ return det1.getRect().x < det2.getRect().x; });
            value = join(valueDets);

            cout << valueDets.size() << endl;
            cout << value << endl;
            cout << (value == name) << endl;

            rectangle(img, keyRect, Scalar(255, 255, 0));
            rectangle(img, valueRect, Scalar(0, 0, 255));
        } else if (key == "VehicleModel") {

        } else if (key == "EngineDisplacement") {

        } else if (key =="DateOfManufacture") {

        } else if (key == "Paint") {

        }

        infoTable[key] = value;
    }

    cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
    cv::imshow("result", img);

    cv::waitKey(0);
    return 0;
}