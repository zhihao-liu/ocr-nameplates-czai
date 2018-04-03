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

#define SHOW_FALSE_DETS 0
#define OUTPUT_DETAILS 0

int ROI_X_BORDER = 8;
int ROI_Y_BORDER = 4;
int CHAR_X_BORDER = 2;
int CHAR_Y_BORDER = 1;


class LeastSquare {
private:
    double a, b;
public:
    LeastSquare(std::vector<double> const& x, std::vector<double> const& y) {
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
    double constant() { return b; };
};

std::vector<std::string> readClassNames(std::string const& path) {
    std::ifstream file(path);
    std::vector<std::string> classNames;
    classNames.push_back("__background__");

    std::string line;
    while (getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
};

template<typename T, typename F>
T const& findMedian(std::vector<T> vec, F const& comp) {
    std::nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end(), comp);
    return vec.at(vec.size() / 2);
};

template<typename T, typename F>
T const& computeMean(std::vector<T> const& vec, F const& map) {
    double avg = 0;
    for (auto item: vec) {
        avg += map(item);
    }
    return avg / vec.size();
};

int xMid(cv::Rect const& rect) {
    return rect.x + round(rect.width / 2.0);
}

int yMid(cv::Rect const& rect) {
    return rect.y + round(rect.height / 2.0);
}

void printAll(std::unordered_map<std::string, std::string> const& map) {
    for (auto const& pair: map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
};

void printAll(std::vector<Detection> const& dets) {
    for (auto const& det: dets) {
        std::cout << det.getClass() << ": " << det.getScore()
             << " x" << det.getRect().x << " y" << det.getRect().y
             << " w" << det.getRect().width << " h" << det.getRect().height << std::endl;
    }
}

void sortInPosition(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2){ return xMid(det1.getRect()) < xMid(det2.getRect()); });
}

bool isSortedInPosition(std::vector<Detection> const& dets) {
    return std::is_sorted(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2){ return xMid(det1.getRect()) < xMid(det2.getRect()); });
}

void imrotate(cv::Mat& img, cv::Mat& newImg, double angleInDegree){
    cv::Point2f pt(img.cols / 2.0, img.rows / 2.0);
    cv::Mat r = getRotationMatrix2D(pt, angleInDegree, 1.0);
    warpAffine(img, newImg, r , img.size());
}

std::string joinDetectedChars(std::vector<Detection> const& dets) {
    assert(isSortedInPosition(dets));

    std::string str = "";
    for (auto const& det: dets) {
        str += det.getClass();
    }
    return str;
}

bool containsConfidentOne(Detection const& det1, Detection const& det2) {
    return (det1.getClass() == "1" && det2.getClass() != "7" && det2.getClass() != "4" && det2.getClass() != "H")
           || (det2.getClass() == "1" && det1.getClass() != "7" && det1.getClass() != "4" && det1.getClass() != "H" );
}

void eliminateHorizontalShifts(std::vector<Detection> &dets) {
    if (dets.empty()) return;

    // remove those lies off the horizontal reference line
    int heightRef = findMedian(dets, [](Detection const &det1, Detection const &det2) {
        return det1.getRect().height < det2.getRect().height;
    }).getRect().height;

    int yMidRef = yMid(findMedian(dets, [](Detection const &det1, Detection const &det2) {
        return yMid(det1.getRect()) < yMid(det2.getRect());
    }).getRect());

    for (auto itr = dets.begin(); itr != dets.end();) {
        if (abs(yMid(itr->getRect()) - yMidRef) > 0.25 * heightRef) {
            itr = dets.erase(itr);
        } else {
            ++itr;
        }
    }
}

int computeXOverlap(cv::Rect const &rect1, cv::Rect const &rect2) {
    int xMin = std::min(rect1.x, rect2.x);
    int xMax = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
    int xOverlap = (rect1.width + rect2.width) - (xMax - xMin);

    return xOverlap <= 0 ? 0 : xOverlap;
}

int computeYOverlap(cv::Rect const &rect1, cv::Rect const &rect2) {
    int yMin = std::min(rect1.y, rect2.y);
    int yMax = std::max(rect1.y + rect1.height, rect2.y + rect2.height);
    int yOverlap = (rect1.height + rect2.height) - (yMax - yMin);

    return yOverlap <= 0 ? 0 : yOverlap;
}

int computeAreaIntersection(cv::Rect const &rect1, cv::Rect const &rect2) {
    int xOverlap = computeXOverlap(rect1, rect2);
    int yOverlap = computeYOverlap(rect1, rect2);

    return xOverlap * yOverlap;
}

double computeIou(cv::Rect const &rect1, cv::Rect const &rect2) {
    int areaIntersection = computeAreaIntersection(rect1, rect2);
    return double(areaIntersection) / (rect1.area() + rect2.area() - areaIntersection);
}

void eliminateOverlaps(std::vector<Detection> &dets) {
    if (dets.size() <= 1) return;
    assert(isSortedInPosition(dets));

    for (auto itr = dets.begin() + 1; itr != dets.end();) {
        // set larger tolerance for "1" because it is overlapped most of the time
        double firstTol = containsConfidentOne(*(itr - 1), *itr) ? 0.6 : 0.4;
        double secondTol = containsConfidentOne(*(itr - 1), *itr) ? 0.6 : 0.3;

        double overlap = computeIou((itr - 1)->getRect(), itr->getRect());
//        std::cout << "iou=" << overlap << std::endl;
        if (overlap > firstTol) {
            if ((itr - 1)->getScore() < itr->getScore()) {
                itr = dets.erase(itr - 1) + 1;
                continue;
            } else {
                itr = dets.erase(itr);
                continue;
            }
        } else if (overlap > secondTol) {
            if ((itr - 1)->getScore() < itr->getScore() && (itr - 1)->getScore() < 0.2) {
                itr = dets.erase(itr - 1) + 1;
                continue;
            } else if (itr->getScore() < (itr - 1)->getScore() && itr->getScore() < 0.2) {
                itr = dets.erase(itr);
                continue;
            }
        };

        ++itr;
    }
}

cv::Rect computeExtent(std::vector<Detection> const &dets) {
    int left = INT_MAX, right = INT_MIN, top = INT_MAX, bottom = INT_MIN;
    for (auto det: dets) {
        if (det.getRect().x < left) left = det.getRect().x;
        if (det.getRect().x + det.getRect().width > right) right = det.getRect().x + det.getRect().width;
        if (det.getRect().y < top) top = det.getRect().y;
        if (det.getRect().y + det.getRect().height > bottom) bottom = det.getRect().y + det.getRect().height;
    }

    return cv::Rect(left, top, right - left, bottom - top);
}

double computeCharAlignmentSlope(std::vector<Detection> const& dets) {
    std::vector<double> xCoords, yCoords;
    std::transform(dets.begin(), dets.end(), back_inserter(xCoords), [](Detection det){ return xMid(det.getRect()); });
    std::transform(dets.begin(), dets.end(), back_inserter(yCoords), [](Detection det){ return yMid(det.getRect()); });

    LeastSquare ls(xCoords, yCoords);
    return ls.slope();
}

cv::Rect& expandRoi(cv::Rect& roi, std::vector<Detection> const& dets) {
    assert(isSortedInPosition(dets));

    int vacancy = 17 - int(dets.size());
    if (vacancy <= 0) return roi;

    double charWidth = computeExtent(dets).width / dets.size();
    double additionalWidth = charWidth * vacancy * 1.1;

    int newX = roi.x;
    int newY = roi.y;
    int newW = round(roi.width + additionalWidth);
    int newH = roi.height;

    double slope = computeCharAlignmentSlope(dets);

    if (slope > 0) {
        newH += round(slope * additionalWidth);
    } else {
        newY += round(slope * additionalWidth);
    }

    roi.x = newX;
    roi.y = newY;
    roi.width = newW;
    roi.height = newH;

    return roi;
}

bool isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent) {
    return (roi.width - detsExtent.width > 2.5 * ROI_X_BORDER) || (roi.height - detsExtent.height > 2.5 * ROI_Y_BORDER);
}

cv::Rect& adjustRoi(cv::Rect& roi, cv::Rect const& detsExtent) {
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

void validateWindow(cv::Rect& window, int width, int height) {
    if (window.x < 0) {
        window.x = 0;
    } else if (window.x >= width) {
        window.x = width - 1;
    }

    if (window.y < 0) {
        window.y = 0;
    } else if (window.y >= height) {
        window.y = height - 1;
    }

    if (window.width < 1) {
        window.width = 1;
    } else if (window.x + window.width > width) {
        window.width = width - window.x;
    }

    if (window.height < 1) {
        window.height = 1;
    } else if (window.y + window.height > height) {
        window.height = height - window.y;
    }
}

void validateWindow(cv::Rect& roi, cv::Mat const& img) {
    // ensure the roi is within the extent of the image after adjustments
    validateWindow(roi, img.cols, img.rows);
}

void validateWindow(cv::Rect& roi, cv::Rect const& extent) {
    // ensure the roi is within the extent of the image after adjustments
    validateWindow(roi, extent.width, extent.height);
}

int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2) {
    return std::abs(xMid(rect1) - xMid(rect2));
}

int estimateCharSpacing(std::vector<Detection> const& dets) {
    assert(isSortedInPosition(dets));

    std::vector<int> spacings;
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        spacings.push_back(computeSpacing((itr - 1)->getRect(), itr->getRect()));
    }

    return findMedian(spacings, std::less<int>());
}

void addGapDetections(PVADetector& detector, std::vector<Detection>& dets, cv::Rect const& roi, cv::Mat const& img) {
    if (dets.empty() || dets.size() >= 17) return;
    assert(isSortedInPosition(dets));

    int spacingRef = estimateCharSpacing(dets);
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        cv::Rect leftRect = (itr - 1)->getRect();
        cv::Rect rightRect = itr->getRect();
        if (computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.x + leftRect.width - CHAR_X_BORDER;
            int gapY = (leftRect.y + rightRect.y) / 2 - CHAR_Y_BORDER;
            int gapW = (rightRect.x - (leftRect.x + leftRect.width)) + CHAR_X_BORDER * 2;
            int gapH = (leftRect.height + rightRect.height) / 2 + CHAR_Y_BORDER * 2;

            int gapXReal = roi.x + gapX;
            int gapYReal = roi.y + gapY;

            cv::Rect gapWindow;
            gapWindow.x = gapXReal;
            gapWindow.y = gapYReal;
            gapWindow.width = gapW;
            gapWindow.height = gapH;

            validateWindow(gapWindow, img);
            std::vector<Detection> gapDets = detector.detect(img(gapWindow));

            if (!gapDets.empty()) {
                gapWindow.x = gapX;
                gapWindow.y = gapY;
                validateWindow(gapWindow, roi);

                Detection& gapDet = gapDets.front();
                gapDet.setRect(gapWindow);

                dets.push_back(gapDet);
            }
        }
    }
}

//void appendTrailDetections(PVADetector &detector, std::vector<Detection> &dets, cv::Rect const &roi, cv::Mat const &img) {
//    if (dets.empty() || dets.size() >= 17) return;
//    assert(isSortedInPosition(dets));
//
//    int vacancy = 17 - int(dets.size());
//
//    int charWidth = computeExtent(dets).width / dets.size();
//    int charHeight = findMedian(dets, [](Detection const& det1, Detection const& det2){ return det1.getRect().height < det2.getRect().height; })
//            .getRect().height;
//    double slope = computeCharAlignmentSlope(dets);
//
//    cv::Rect lastWindow = dets.back().getRect();
//    int windowX = lastWindow.x + lastWindow.width - CHAR_X_BORDER;
//    int windowY = lastWindow.y + charWidth * slope;
//    int windowW = charWidth + CHAR_X_BORDER * 2;
//    int windowH = charHeight + CHAR_Y_BORDER * 2;
//
//    for (int i = 0; i < vacancy; ++i) {
//        int windowXReal = roi.x + windowX;
//        int windowYReal = roi.y + windowY;
//
//        cv::Rect trailWindow;
//        trailWindow.x = windowXReal;
//        trailWindow.y = windowYReal;
//        trailWindow.width = windowW;
//        trailWindow.height = windowH;
//
//        validateWindow(trailWindow, img);
//        std::vector<Detection> trailDets = detector.detect(img(trailWindow));
//
//        if (!trailDets.empty()) {
//            trailWindow.x = windowX;
//            trailWindow.y = windowY;
//            validateWindow(trailWindow, roi);
//
//            Detection& trailDet = trailDets.front();
//            trailDet.setRect(trailWindow);
//
//            dets.push_back(trailDet);
//        }
//
//        // update for the next possible trailing position
//        windowX = windowX + windowW - CHAR_X_BORDER;
//        windowY = windowY + charWidth * slope;
//    }
//}

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    namespace fs = boost::filesystem;

    string pathInputDir = "/home/cuizhou/lzh/data/raw-alfaromeo/";
//    string pathInputDir = "/home/cuizhou/lzh/data/selected-test/";

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

    int countAll = 0, countCorrect = 0, countShorter = 0, countLonger= 0, countWrong = 0;

    for (fs::directory_iterator itr(pathInputDir); itr != fs::directory_iterator(); ++itr) {
        string pathImg = itr->path().string();
//        string pathImg = "/home/cuizhou/lzh/data/raw-alfaromeo/ZAREAEBN1H7545545.jpg";
        string fileName = itr->path().filename().string();
        string id = fileName.substr(0, fileName.length() - 4);

        Mat img = imread(pathImg);
        if (img.empty()) continue;

//        unordered_map<string, string> infoTable;

        vector<Detection> keyDets = detectorKeys.detect(img);
        for (auto const& keyDet: keyDets) {
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
                eliminateHorizontalShifts(valueDets);
                eliminateOverlaps(valueDets);

                double slope = computeCharAlignmentSlope(valueDets);
                if (std::abs(slope) > 0.025) {
                    double angle = std::atan(slope) / CV_PI * 180;
                    imrotate(img, img, angle);

                    vector<Detection> keyDets = detectorKeys.detect(img);
                    for (auto const& keyDet: keyDets) {
                        if (keyDet.getClass() == "VehicleId") {
                            keyRect = keyDet.getRect();
                        }
                    }

                    valueRect = cv::Rect(keyRect.x + keyRect.width + 5, round(keyRect.y - keyRect.height * 0.25),round(keyRect.width * 1.75), round(keyRect.height * 1.5));
                    valueDets = detectorValues.detect(img(valueRect));

                    sortInPosition(valueDets);
                    eliminateHorizontalShifts(valueDets);
                    eliminateOverlaps(valueDets);
                }

                // second round in the network
                adjustRoi(valueRect, computeExtent(valueDets));
                expandRoi(valueRect, valueDets);
                validateWindow(valueRect, img);
                valueDets = detectorValues.detect(img(valueRect));

                sortInPosition(valueDets);
                eliminateHorizontalShifts(valueDets);
                eliminateOverlaps(valueDets);

                Rect detsExtent = computeExtent(valueDets);
                if (isRoiTooLarge(valueRect, detsExtent)) {
                    // third round in the network
                    adjustRoi(valueRect, detsExtent);
                    validateWindow(valueRect, img);
                    valueDets = detectorValues.detect(img(valueRect));

                    sortInPosition(valueDets);
                    eliminateHorizontalShifts(valueDets);
                    eliminateOverlaps(valueDets);
                }

                if (valueDets.size() < 17) {
                    // fourth round in the network
                    addGapDetections(detectorValues, valueDets, valueRect, img);
                    sortInPosition(valueDets);
                }

                value = joinDetectedChars(valueDets);

                if (OUTPUT_DETAILS) {
                    printAll(valueDets);
                    cout << valueDets.size() << endl;
                    cout << value << endl;
                }

                ++countAll;
                string msg = value == id ? (++countCorrect, "yes") : (value.length() == id.length() ? (++countWrong, "wrong") : (value.length() > id.length()) ? (++countLonger, "longer") : (++countShorter, "shorter"));
                cout << countAll << " -- " << msg << endl;

                // show those with incorrect results
                if ( value != id) {
                    Mat dst = img.clone();
                    Mat roi = dst(valueRect);
                    detectorValues.drawBox(roi, valueDets);
                    rectangle(dst, keyRect, Scalar(255, 255, 255));
                    rectangle(dst, valueRect, Scalar(255, 255, 255));

                    imwrite("/home/cuizhou/lzh/data/results/" + fileName, dst);

                    if (SHOW_FALSE_DETS) {
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

        cout << countCorrect << " out of " << countAll << " (" << 100 * float(countCorrect) / countAll << "%) correct." << "...";
        cout << countWrong << " out of " << countAll << " (" << 100 * float(countWrong) / countAll << "%) wrong." << "...";
        cout << countShorter << " out of " << countAll << " (" << 100 * float(countShorter) / countAll << "%) shorter." << "...";
        cout << countLonger << " out of " << countAll << " (" << 100 * float(countLonger) / countAll << "%) longer." << endl;
    }

    waitKey(0);
    return 0;
}