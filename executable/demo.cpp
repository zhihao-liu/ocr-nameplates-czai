//
// Created by Zhihao Liu on 18-3-21.
//

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pvaDetector.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;


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

template<typename T, typename F>
T const& medianWithMutation(vector<T>& vec, F const &comp) {
    nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end(), comp);
    return vec[vec.size() / 2];
};

int xMid(Rect const &rect) {
    return rect.x + round(rect.width / 2.0);
}

int yMid(Rect const &rect) {
    return rect.y + round(rect.height / 2.0);
}

string joinDetectedChars(vector<Detection> const& dets) {
    string str = "";
    for (auto const& det: dets) {
        str += det.getClass();
    }
    return str;
}

void printAll(unordered_map<string, string> const& map) {
    for (auto const& pair: map) {
        cout << pair.first << ": " << pair.second << endl;
    }
};

void printAll(vector<Detection> const& dets) {
    for (auto const& det: dets) {
        cout << det.getClass() << ": " << det.getScore() << endl;
    }
}

void sortInPosition(vector<Detection>& dets) {
    sort(dets, [](Detection const& det1, Detection const& det2){ return det1.getRect().x < det2.getRect().x; });
}

void validateVehicleId(vector<Detection>& dets) {
    if (dets.size() <= 17) return;

    // remove those lies off the horizontal reference line
    int heightRef = medianWithMutation(dets, [](Detection const &det1, Detection const &det2) {
        return det1.getRect().height < det2.getRect().height;
    }).getRect().height;

    int yMidRef = yMid(medianWithMutation(dets, [](Detection const &det1, Detection const &det2) {
        return yMid(det1.getRect()) < yMid(det2.getRect());
    }).getRect());

    for (auto det = dets.begin(); det != dets.end();) {
        if (abs(yMid(det->getRect()) - yMidRef) > 0.25 * heightRef) {
            dets.erase(det);
        } else {
            ++det;
        }
    }
    if (dets.size() <= 17) return;

    // remove overlapped detections according to scores
    int widthRef = medianWithMutation(dets, [](Detection const &det1, Detection const &det2) {
        return det1.getRect().width < det2.getRect().width;
    }).getRect().width;

    sortInPosition(dets);

    for (auto det = dets.begin() + 1; det != dets.end();) {
        if (abs(xMid((det - 1)->getRect()) - xMid(det->getRect())) < 0.5 * widthRef) {
            dets.erase((det - 1)->getScore() < det->getScore() ? det - 1 : det);
        } else {
            ++det;
        }
    }
    if (dets.size() <= 17) return;
}

int main(int argc, char* argv[]) {
    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo/";

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
    detectorValues.setThresh(0.1, 0.3);
    detectorValues.setComputeMode("gpu", 0);

    int countAll = 0, countCorrect = 0;

    auto itrEnd = fs::directory_iterator();
    for (fs::directory_iterator itr(pathInputDir); itr != itrEnd; ++itr) {
        string pathImg = itr->path().string();
        string fileName = itr->path().filename().string();
        string id = fileName.substr(0, fileName.length() - 4);

        Mat img = imread(pathImg);
        if (img.empty()) continue;

//        unordered_map<string, string> infoTable;

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
                Rect valueRect(round(keyRect.x + keyRect.width * 1.06), round(keyRect.y - keyRect.height * 0.25),round(keyRect.width * 1.4), round(keyRect.height * 1.5));
                vector<Detection> valueDets = detectorValues.detect(img(valueRect));

                validateVehicleId(valueDets);
                sortInPosition(valueDets);
                value = joinDetectedChars(valueDets);

//                printAll(valueDets);
//                cout << valueDets.size() << endl;
//                cout << value << endl;
                ++countAll;
                countCorrect += value == id;
                string msg = value == id ? "yes" : (value.length() == id.length() ? "wrong" : (value.length() > id.length()) ? "longer" : "shorter");
                cout << msg << endl;

                // show those with incorrect results
//                if (value != id) {
//                    cout << value << endl;
//
//                    Mat dst = img.clone();
//                    Mat roi = dst(valueRect);
//                    detectorValues.drawBox(roi, valueDets);
//                    rectangle(dst, keyRect, Scalar(255, 255, 255));
//                    rectangle(dst, valueRect, Scalar(255, 255, 255));
//
////                    imwrite("/home/cuizhou/lzh/data/results/" + fileName, dst);
//
//                    cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
//                    cv::imshow("result", dst);
//                    cv::namedWindow("source", CV_WINDOW_AUTOSIZE);
//                    cv::imshow("source", img);
//                    waitKey(0);
//                }

//                rectangle(img, keyRect, Scalar(255, 255, 0));
//                rectangle(img, valueRect, Scalar(0, 0, 255));
            } else if (key == "VehicleModel") {

            } else if (key == "EngineDisplacement") {

            } else if (key =="DateOfManufacture") {

            } else if (key == "Paint") {

            }

//            infoTable[key] = value;
        }
    }

    cout << countCorrect << " out of " << countAll << " correct." << endl;

    waitKey(0);
    return 0;
}