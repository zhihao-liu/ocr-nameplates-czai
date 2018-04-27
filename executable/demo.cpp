//
// Created by Zhihao Liu on 18-3-21.
//

#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "detector.h"
#include "ocr_interface.hpp"
#include "ocr_aux/detection_proc.h"


int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    using namespace cuizhou;
    using namespace boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/corrected-volkswagen";

    string dirPvaValues = "/home/cuizhou/lzh/models/models_volkswagen/pva_values/";
    string ptPvaValues = dirPvaValues + "merge_svd.prototxt";
    string modelPvaValues = dirPvaValues + "nameplate_dz_iter_100000_merge_svd.caffemodel";
    vector<string> classesPvaValues = readClassNames(dirPvaValues + "predefined_classes.txt");

    Detector detectorValues;
    detectorValues.init(ptPvaValues, modelPvaValues, classesPvaValues);
    detectorValues.setComputeMode("gpu", 0);

    OcrInterface ocr(OcrType::NAMEPLATE_VOLKSWAGEN, {detectorValues});

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
//        cout << ocr.getResultAsString() << endl;

//        imshow("", ocr.drawResult());
        imshow("", ocr.image());
        waitKey(0);
    }

    auto end = std::chrono::system_clock::now();

    cout << "Time elapsed: " << (end - start).count() << endl;

    return 0;
}