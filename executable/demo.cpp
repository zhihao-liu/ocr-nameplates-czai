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

#define SHOW_RESULT false
#define OUTPUT_DETAILS false

int const WIDTH_CHAR = 14;
int const HEIGHT_CHAR = 28;

int ROI_X_BORDER = 10;
int ROI_Y_BORDER = 5;


class LeastSquare {
private:
    double a, b;
public:
    LeastSquare(vector<double> const& x, vector<double> const& y) {
        double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
        for(int i = 0; i < x.size(); ++i)
        {
            t1 += x[i] * x[i];
            t2 += x[i];
            t3 += x[i] * y[i];
            t4 += y[i];
        }
        a = (t3*x.size() - t2*t4) / (t1*x.size() - t2*t2);
        b = (t1*t4 - t2*t3) / (t1*x.size() - t2*t2);
    }

    double slope() { return a; };
    double constant() {return b; };
};

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
T const& findMedian(vector<T> vec, F const& comp) {
    nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end(), comp);
    return vec.at(vec.size() / 2);
};

template<typename T, typename F>
T const& computeMean(vector<T> const& vec, F const& map) {
    double avg = 0;
    for (auto item: vec) {
        avg += map(item);
    }
    return avg / vec.size();
};

int xMid(Rect const& rect) {
    return rect.x + round(rect.width / 2.0);
}

int yMid(Rect const& rect) {
    return rect.y + round(rect.height / 2.0);
}

void printAll(unordered_map<string, string> const& map) {
    for (auto const& pair: map) {
        cout << pair.first << ": " << pair.second << endl;
    }
};

void printAll(vector<Detection> const& dets) {
    for (auto const& det: dets) {
        cout << det.getClass() << ": " << det.getScore() << " x" << xMid(det.getRect()) << endl;
    }
}

void sortInPosition(vector<Detection>& dets) {
    sort(dets, [](Detection const& det1, Detection const& det2){ return xMid(det1.getRect()) < xMid(det2.getRect()); });
}

bool isSortedInPosition(vector<Detection> const& dets) {
    return is_sorted(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2){ return xMid(det1.getRect()) < xMid(det2.getRect()); });
}

string joinDetectedChars(vector<Detection> const& dets) {
    assert(isSortedInPosition(dets));

    string str = "";
    for (auto const& det: dets) {
        str += det.getClass();
    }
    return str;
}

bool containsConfidentOne(Detection const& det1, Detection const& det2) {
    return (det1.getClass() == "1" && det2.getClass() != "7" && det2.getClass() != "4" && det2.getClass() != "H")
           || (det2.getClass() == "1" && det1.getClass() != "7" && det1.getClass() != "4" && det1.getClass() != "H" );
}

void eliminateYOutliers(vector<Detection>& dets) {
    if (dets.size() == 0) return;

    // remove those lies off the horizontal reference line
    int heightRef = findMedian(dets, [](Detection const &det1, Detection const &det2) {
        return det1.getRect().height < det2.getRect().height;
    }).getRect().height;

    int yMidRef = yMid(findMedian(dets, [](Detection const &det1, Detection const &det2) {
        return yMid(det1.getRect()) < yMid(det2.getRect());
    }).getRect());

    for (auto det = dets.begin(); det != dets.end();) {
        if (abs(yMid(det->getRect()) - yMidRef) > 0.25 * heightRef) {
            det = dets.erase(det);
        } else {
            ++det;
        }
    }
}

void eliminateXOverlaps(vector<Detection>& dets) {
    if (dets.size() <= 1) return;
    assert(isSortedInPosition(dets));

    // remove overlapped detections according to scores
    vector<int> spacings;
    for (auto det = dets.begin() + 1; det != dets.end(); ++det) {
        spacings.push_back(abs(xMid((det - 1)->getRect()) - xMid(det->getRect())));
    }
    int spacingRef = findMedian(spacings, less<int>());

    for (auto det = dets.begin() + 1; det != dets.end();) {
        // set larger tolerance for "1" because it is overlapped most of the time
        double firstTol = containsConfidentOne(*(det - 1), *det) ? 0.2 : 0.4;
        double secondTol = containsConfidentOne(*(det - 1), *det) ? 0.2 : 0.7;

        int xSpacing = abs(xMid((det - 1)->getRect()) - xMid(det->getRect()));
        if (xSpacing < firstTol * spacingRef) {
            if ((det - 1)->getScore() < det->getScore()) {
                det = dets.erase(det - 1) + 1;
                continue;
            } else {
                det = dets.erase(det);
                continue;
            }
        } else if (xSpacing < secondTol * spacingRef) {
            if ((det - 1)->getScore() < det->getScore() && (det - 1)->getScore() < 0.2) {
                det = dets.erase(det - 1) + 1;
                continue;
            } else if (det->getScore() < (det - 1)->getScore() && det->getScore() < 0.2) {
                det = dets.erase(det);
                continue;
            }
        };

        ++det;
    }
}

Rect computeExtent(vector<Detection> const &dets) {
    int left = INT_MAX, right = INT_MIN, top = INT_MAX, bottom = INT_MIN;
    for (auto det: dets) {
        if (det.getRect().x < left) left = det.getRect().x;
        if (det.getRect().x + det.getRect().width > right) right = det.getRect().x + det.getRect().width;
        if (det.getRect().y < top) top = det.getRect().y;
        if (det.getRect().y + det.getRect().height > bottom) bottom = det.getRect().y + det.getRect().height;
    }

    return Rect(left, top, right - left, bottom - top);
}

Rect& expandRoi(Rect& roi, vector<Detection> dets) {
    assert(isSortedInPosition(dets));
    int vacancy = dets.size() < 17 ? 17 - dets.size() : 0;

    int newX = roi.x;
    int newWidth = roi.width + vacancy * WIDTH_CHAR;
    int newY = roi.y;
    int newHeight = roi.height;

    vector<double> xCoords, yCoords;
    transform(dets.begin(), dets.end(), back_inserter(xCoords), [](Detection det){ return xMid(det.getRect()); });
    transform(dets.begin(), dets.end(), back_inserter(yCoords), [](Detection det){ return yMid(det.getRect()); });

    LeastSquare ls(xCoords, yCoords);
    double slope = ls.slope();

    if (slope > 0) {
        newHeight += slope * vacancy * WIDTH_CHAR;
    } else {
        newY += slope * vacancy * WIDTH_CHAR;
    }

    roi.x = newX;
    roi.y = newY;
    roi.width = newWidth;
    roi.height = newHeight;
}

bool isRoiTooLarge(Rect const& roi, Rect const& detsExtent) {
    return (roi.width - detsExtent.width > 2.5 * ROI_X_BORDER) || (roi.height - detsExtent.height > 2.5 * ROI_Y_BORDER);
}

Rect& adjustRoi(Rect& roi, Rect const& detsExtent) {
    int newLeft = roi.x + (detsExtent.x - ROI_X_BORDER);
    int newRight = roi.x + (detsExtent.x + detsExtent.width + ROI_X_BORDER);
    int newTop = roi.y + (detsExtent.y - ROI_Y_BORDER);
    int newBottom = roi.y + (detsExtent.y + detsExtent.height + ROI_Y_BORDER);

    roi.x = newLeft;
    roi.y = newTop;
    roi.width = newRight - newLeft;
    roi.height = newBottom - newTop;

    return roi;
}

Rect& validateRoi(Rect& roi, Mat const& img) {
    // ensure the roi is within the extent of the image after adjustments
    if (roi.x < 0) roi.x = 0;
    if (roi.x + roi.width > img.cols) roi.width = img.cols - roi.x;
    if (roi.y < 0) roi.y = 0;
    if (roi.y + roi.height > img.rows) roi.height = img.rows - roi.y;
    return roi;
}

void updateWithDetectionsGray(vector<Detection>& dets, vector<Detection> const& detsGray) {
    if (dets.size() != detsGray.size()) return;

    assert(isSortedInPosition(dets) && isSortedInPosition(detsGray));
    for (int i = 0; i < dets.size(); ++i) {
        if (detsGray[i].getScore() - dets[i].getScore() > 0.3) {
            dets[i] = detsGray[i];
        }
    }
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
//        string pathImg = "/home/cuizhou/lzh/data/selected-test/ZAREAECN3H7544296.jpg";
        string fileName = itr->path().filename().string();
        string id = fileName.substr(0, fileName.length() - 4);

        Mat img = imread(pathImg);
        if (img.empty()) continue;

//        unordered_map<string, string> infoTable;

        vector<Detection> keyDets = detectorKeys.detect(img);
        for (auto keyDet: keyDets) {
            string key = keyDet.getClass();
            Rect keyRect = keyDet.getRect();
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
                // first round in the network
                Rect valueRect(keyRect.x + keyRect.width + 5, round(keyRect.y - keyRect.height * 0.25),round(keyRect.width * 1.75), round(keyRect.height * 1.5));
                vector<Detection> valueDets = detectorValues.detect(img(valueRect));

                sortInPosition(valueDets);
                eliminateYOutliers(valueDets);
                eliminateXOverlaps(valueDets);

                // second round in the network
                adjustRoi(valueRect, computeExtent(valueDets));
                expandRoi(valueRect, valueDets);
                validateRoi(valueRect, img);
                valueDets = detectorValues.detect(img(valueRect));

                sortInPosition(valueDets);
                eliminateYOutliers(valueDets);
                eliminateXOverlaps(valueDets);


                Rect detsExtent = computeExtent(valueDets);
                if (isRoiTooLarge(valueRect, detsExtent)) {
                    // third round in the network
                    adjustRoi(valueRect, detsExtent);
                    validateRoi(valueRect, img);
                    valueDets = detectorValues.detect(img(valueRect));

                    sortInPosition(valueDets);
                    eliminateYOutliers(valueDets);
                    eliminateXOverlaps(valueDets);
                }

//                Mat valueRoiGray = img(valueRect).clone();
//                cvtColor(valueRoiGray, valueRoiGray,CV_RGB2GRAY);
//                cv::adaptiveThreshold(valueRoiGray, valueRoiGray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 0);
//                cvtColor(valueRoiGray, valueRoiGray,CV_GRAY2RGB);
//                vector <Detection> valueDetsGray = detectorValues.detect(valueRoiGray);
//                eliminateYOutliers(valueDetsGray);
//                eliminateXOverlaps(valueDetsGray);
//
//                sortInPosition(valueDets);
//                sortInPosition(valueDetsGray);
//                updateWithDetectionsGray(valueDets, valueDetsGray);


                value = joinDetectedChars(valueDets);

                if (OUTPUT_DETAILS) {
                    printAll(valueDets);
                    cout << valueDets.size() << endl;
                    cout << value << endl;
                }

                ++countAll;
                countCorrect += value == id;
                string msg = value == id ? "yes" : (value.length() == id.length() ? "wrong" : (value.length() > id.length()) ? "longer" : "shorter");
                cout << countAll << " -- " << msg << endl;

                // show those with incorrect results
                if (value != id) {
                    Mat dst = img.clone();
                    Mat roi = dst(valueRect);
                    detectorValues.drawBox(roi, valueDets);
                    rectangle(dst, keyRect, Scalar(255, 255, 255));
                    rectangle(dst, valueRect, Scalar(255, 255, 255));

                    imwrite("/home/cuizhou/lzh/data/results/" + fileName, dst);

                    if (SHOW_RESULT) {
                        cout << "Ref: " << id << endl;
                        cout << "Out: " << value << endl;
                        cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
                        cv::imshow("result", dst);
                        cv::namedWindow("source", CV_WINDOW_AUTOSIZE);
                        cv::imshow("source", img);
                        waitKey(0);
                    }
                }
            } else if (key == "VehicleModel") {

            } else if (key == "EngineDisplacement") {

            } else if (key =="DateOfManufacture") {

            } else if (key == "Paint") {

            }
//            infoTable[key] = value;
        }
    }

    cout << countCorrect << " out of " << countAll << " (" << 100 * float(countCorrect) / countAll << "%) correct." << endl;

    waitKey(0);
    return 0;
}