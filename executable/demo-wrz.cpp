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
#include <math.h>
#include <string>
#include <dirent.h>
#include <algorithm>
// googlenet
#include "googlenet/classifier.h"

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

int computeAreaOverlap(cv::Rect const &rect1, cv::Rect const &rect2) {
    int xOverlap = computeXOverlap(rect1, rect2);
    int yOverlap = computeYOverlap(rect1, rect2);

    return xOverlap * yOverlap;
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

// iou 去重
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

// Gap :dets 是排序后的结果
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



// ---------------- wurui added ------------------------//
using namespace cv;
using namespace std;

void cropImg(Mat &input){
    Mat subimg(642,1058,CV_8UC3,Scalar(0,0,0));
    int w = input.cols;
    int h = input.rows;
    int nw,nh;
    float ratio = 0.0;
    if ((float)w/(float)h < 1056.0/640.0){
        // h is main
        ratio = 640.0/h;
        nh = 640;
        nw = (int)(ratio*w);
        cv::resize(input,input,Size(nw,nh));
//        cout << "528-nw/2   "<< 528-nw/2 << endl;
        Rect roi(528-nw/2,0,nw,nh);
        input.copyTo(subimg(roi));
    } else {
        // w is main
        ratio = 1056.0/w;
//        cout << "ratio "<< ratio << endl;
        nh = (int)(h*ratio);
        nw = 1056;
        cv::resize(input,input,Size(nw,nh));
//        cout << "320-nh/2  "<< 320-nh/2 << endl;
        Rect roi(0,max(0,320-nh/2),nw,nh);
//        cout << roi.x << " " << roi.y << " "  << roi.width << " " << roi.height << endl;
//        cout << input.cols << " " << input.rows << endl;
        input.copyTo(subimg(roi));
    }
    subimg.copyTo(input);
}

void getFileNames(std::string path, vector<std::string> &files)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            files.push_back(string(ptr->d_name));
        else if(ptr->d_type == 10)    ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push_back(string(ptr->d_name));
            /*
                memset(base,'\0',sizeof(base));
                strcpy(base,basePath);
                strcat(base,"/");
                strcat(base,ptr->d_nSame);
                readFileList(base);
            */
        }
    }
    closedir(dir);
    //排序，按从小到大排序
    sort(files.begin(), files.end());
    for(int i=0; i<files.size(); ++i)
        files[i] = path + files[i];
}
void roiMassPower(const Rect &roiMass, const Rect &roiPower, Rect &roi){
    int ymin = min(roiMass.y, roiPower.y);
    int ymax = max(roiMass.y+roiMass.height,roiPower.y+roiPower.height);
    int h_iou = max(0,roiMass.height+roiPower.height-(ymax-ymin));
    int x1,x2,y1,y2;
    if (h_iou>0){
        x1 = roiMass.x+roiMass.width+5;
        y1 = roiMass.y-5;
        x2 = roiPower.x+(int)(roiPower.width*1.5);
        y2 = roiPower.y + roiPower.height+(int)(h_iou*1.2)+15;
    } else{
        x1 = roiMass.x+roiMass.width+5;
        y1 = roiMass.y-5;
        x2 = roiPower.x+(int)(roiPower.width*1.5);
        y2 = roiPower.y + roiPower.height+5;
    }
    roi = Rect(x1,y1,max(0,x2-x1),max(0,y2-y1));
}

bool compareX(Detection &r1, Detection &r2){
    return r1.getRect().x<r2.getRect().x;
}

bool compareY(Detection &r1, Detection &r2){
    return r1.getRect().y<r2.getRect().y;
}

bool compareS(Detection &r1, Detection &r2){
    return r1.getScore()>r2.getScore();
}

bool isNumber(string &str){
//    cout << "isnumber: "<<str.size()<<endl;
    if (str.size()==1) {
//        bool b = str[0]>='0' && str[0]<='9';
//        cout << str << " " << str[0] << " "  <<b <<endl;
        return str[0]>='0' && str[0]<='9';
    }
    else return false;
}

void deleteBoxNotLine(vector<Detection> &detections, vector<Detection> &newdetections){
    if(detections.size()<3) {
        newdetections.clear();
        newdetections.insert(newdetections.end(),detections.begin(),detections.end());
        return;
    }
    int hmean=0;
    for (Detection d:detections){
        hmean+=d.getRect().height;
//        cout << d.getRect().x << " " << d.getRect().y << "cls:"<< d.getClass()<< endl;
    }
    hmean = (int)(hmean/(float)detections.size());
    float thresh = 0.35;
//    cout << "hmean :" << hmean<< " thresh "<<(int)(thresh*hmean)<<endl;

    vector<vector<Detection>> group;
    vector<Detection> center;
    center.push_back(detections[0]);
    for (int i = 0; i < detections.size(); ++i) {
        bool isNew = true;
        for (int j = 0; j < group.size(); ++j) {
            if (abs(group[j][0].getRect().y - detections[i].getRect().y)< (int)(thresh*hmean)) {
//                cout << " dy = " << abs(group[j][0].getRect().y - detections[i].getRect().y) << endl;
                group[j].push_back(detections[i]);
                isNew = false;
            }
        }
        if(isNew){
            vector<Detection> newCenter;
            newCenter.push_back(detections[i]);
            group.push_back(newCenter);
        }
    }

    int index=0;
    int size=group[0].size();
    for (int i = 0; i < group.size(); ++i) {
//        cout << " group size " << group[i].size() << endl;
        if(group[i].size()>size){
            size = group[i].size();
            index=i;
        }
    }
//    cout << " group index "<<index << " size "<< size << endl;
    for (int k = 0; k < group[index].size(); ++k) {
//        cout << " result detection " << group[index][k].getRect().y << " cls:" <<  group[index][k].getClass() <<endl;
        newdetections.push_back(group[index][k]);
    }

//    newdetections.push_back(detections[0]);
//    newdetections.push_back(detections[detections.size()-1]);
//    for (int i = 1; i < detections.size()-1; ++i) {
//        if( abs(detections[i].getRect().y-detections[i-1].getRect().y)>hmean*0.6 && abs(detections[i].getRect().y-detections[i+1].getRect().y)>hmean*0.6){
//            continue;
//        }
//        newdetections.push_back(detections[i]);
//    }
}

//void gapDetect(PVADetector &detectorValues,vector<Detection> &valueDate,cv::Rect &roiDate,Mat &image, float conf=0.1){
//    // valueDate 在 1056 * 640
//    // gap和框的宽度比较
//    sort(valueDate.begin(), valueDate.end(), compareX);
//    if (valueDate.size()<2) return;
//    vector<int> gaps;
//    int sum=0; // 平均宽度
//    int hsum=0; // 平均高度
//    for (int i = 0; i < valueDate.size()-1; ++i) {
//        sum+=valueDate[i].getRect().width;
//        hsum+=valueDate[i].getRect().height;
//        int tmpgap = max(valueDate[i+1].getRect().x - valueDate[i].getRect().x - valueDate[i].getRect().width,0);
//        cout << "gap :" << tmpgap<<endl;
//        gaps.push_back(tmpgap);
//    }
//    sum = (int)((valueDate[valueDate.size()-1].getRect().width+sum)/(float)(valueDate.size()));
//    hsum = (int)((valueDate[valueDate.size()-1].getRect().height+hsum)/(float)(valueDate.size()));
//    cout << " mean width "<< sum << " h" << hsum <<endl;
//
//    float thresh = 0.35;
//    vector<int> index;// = -1;
//    for (int j = 0; j < gaps.size(); ++j) {
//        if((float)(gaps[j])>thresh*sum) index.push_back(j);
//    }
//    // debug
//    Mat imgtmp = image.clone();
//    vector<Rect> gaproi;
//    for (int k = 0; k < index.size(); ++k) {
//        cout << " gap index "<< index[k]<<endl;
//        int x1,y,x2,h;
//        x1 =roiDate.x+ valueDate[index[k]].getRect().x+valueDate[index[k]].getRect().width - 4;
//        y = roiDate.y + valueDate[index[k]].getRect().y -2;
//        x2 =roiDate.x+ valueDate[index[k]+1].getRect().x+4;
//        h = valueDate[index[k]].getRect().height+4;
//        cout <<" roigap " << x1<<" "<<y<<" "<<abs(x2-x1)<<" "<<h<< endl;
//        Rect roigap(x1,y,abs(x2-x1),h);
//        gaproi.push_back(roigap);
////        rectangle(imgtmp,roigap,Scalar(0,255,0),1);
//    }
//
//    detectorValues.setThresh(conf,0.1);
//    // re detect
//    float hthresh = 0.65;
//    Mat inputimg = image(roiDate).clone();
//    cropImg(inputimg);
//    for (int i = 0; i < gaproi.size(); ++i) {
//        Mat subimg;
//        cout <<"debug1"<<endl;
//
//        inputimg(gaproi[i]).copyTo(subimg);
//        cout <<"debug2"<<endl;
//
//        cropImg(subimg);
//
////        imshow("imdetect", detectImg);
////        int key = cv::waitKey(0);
////        if(key>0) cout<<endl;
//
//        vector<Detection> dgap = detectorValues.detect(subimg);
//        if(dgap.size()<1) return;
//
//        // 高度太小的
//        sort(dgap.begin(),dgap.end(),compareS);
//        cout << "h thresh "<<hthresh*hsum<< endl;
//
//        int nx = (dgap[0].getRect().x-rcDetect.x) +  gaproi[i].x - roiDate.x;
//        int ny = (dgap[0].getRect().y-rcDetect.y) +  gaproi[i].y - roiDate.y;
//        Rect newrect(nx, ny, dgap[0].getRect().width, dgap[0].getRect().height);
//        dgap[0].setRect(newrect);
//        cout << " gap detect result = " << dgap[0].getClass() << endl;
//        valueDate.push_back(dgap[0]);
//
////        for(Detection d:dgap){
////            cout << "gap detection: " << d.getClass() << " s:" << d.getScore() << endl;
////            cout << "rect" << d.getRect().height << endl;
////
////            if ((float)d.getRect().height<hthresh*hsum) continue;
////
//////            rectangle(detectImg,d.getRect(),Scalar(0,255,0),1);
//////            cv::imshow("gap detect", detectImg);
//////            int key = cv::waitKey(0);
//////            if(key>0) cout<<endl;
////        }
//    }
//}

bool compareBig(int a, int b){ return a>b;}

float getiou(Rect &r1, Rect &r2){
    cout<<r1;
    cout<<r2;
    int xs[4] = {r1.x, r2.x, r1.x+r1.width, r2.x+r2.width};
    int ys[4] = {r1.y, r2.y, r1.y+r1.height, r2.y+r2.height};
    vector<int> xxs(xs,xs+4);
    vector<int> yys(ys,ys+4);
    sort(xxs.begin(),xxs.end(),compareBig);
    sort(yys.begin(),yys.end(),compareBig);
    int xmin = xxs[3];
    int ymin = yys[3];
    int xmax = xxs[0];
    int ymax = yys[0];
    cout << xmax << " " << xmin << " " << ymax << " " << ymin << endl;
    int nw = r1.width+r2.width-(xmax-xmin);
    int ny = r1.height+r2.height-(ymax-ymin);
    cout << "nx "<< nw << " ny " << ny << endl;
    if(nw<=0 || ny<=0) return 0;
    else{
        float s = nw*ny*1.0;
        cout << s << endl;
        cout << r1.width*r1.height + r2.height*r2.width<<endl;
        return s/( r1.width*r1.height + r2.height*r2.width - s);
    }
}

void iouOverlap(vector<Detection> &det, float threshiou){
    map<int,bool> isdel;
    for (int i = 0; i < det.size(); ++i) {
        isdel.insert(std::pair<int,bool>(i,false));
    }
    for (int i = 0; i < det.size(); ++i) {
        if (isdel[i]==true) continue;
        for (int j = i+1; j < det.size(); ++j) {
            if (isdel[i]==true) continue;
            Rect r1 = det[i].getRect();
            Rect r2 = det[j].getRect();
            float iou = getiou(r1, r2);

            if (iou>threshiou){
                // merge
                cout <<" iou >>>>>>>>" << iou << endl;
                if(det[i].getScore()>det[j].getScore()){
                    isdel[j]=true;
                } else {
                    isdel[i]=true;
                }
            }
        }
    }
    vector<Detection> result;

    for (int i = 0; i < isdel.size(); ++i) {
        if(isdel[i]==false) result.push_back(det[i]);
    }

    cout << "iou function ==============" <<isdel.size()<<endl;
    det.clear();
    det.insert(det.end(),result.begin(),result.end());
}

// 通用检测流程函数
void subCommonDetectProcess(PVADetector &detectorValues, Mat &imgSrc, Rect &roi, vector<Detection> &detections,
                         bool ifContainEnglish = false, float conf = 0.2, float iouThresh=0.1){
    // 检测流程
    Mat subimg;
    imgSrc(roi).copyTo(subimg);
    cropImg(subimg);
//    imshow("subimg", subimg);
//    waitKey(0);

    detectorValues.setThresh(conf,iouThresh);
    vector<Detection> valueDate = detectorValues.detect(subimg);

    // 筛选数字
    vector<Detection> temp;
    if (ifContainEnglish==false){
        for(Detection d:valueDate){
            string cls = d.getClass();
            if (isNumber(cls)) temp.push_back(d);
        }
    } else{
        temp = valueDate;
    }

    // iou 去重叠
    cout << " before iou overlap "<< temp.size()<< endl;
    if(temp.size()>0) eliminateOverlaps(temp);
//    iouOverlap(temp, 0.5);
    cout << " after iou overlap" << temp.size() << endl;
//    cout << " class: " << temp[0].getClass() << endl;

    // 去除不在一条线上的
    vector<Detection> tempd;
    deleteBoxNotLine(temp,tempd);

    cout << "after delete not in line"<< tempd.size() <<endl;

    // output
    detections.clear();
    detections.insert(detections.end(), tempd.begin(), tempd.end());
}

// 偏移 roi 修正 y
bool roiMove(Mat &imgSrc, vector<Detection> &detections, Rect &roiSrc, Rect &newRoi){
    // detections 的坐标在 1056*640 的图上
    // roiSrc 是在原图上的 roi
    // 返回在原图上的 roi
    int threshy = 100;
    bool needmove = false;
    newRoi = roiSrc;
    if(detections.size()<1) return false;
    int cy;
    for(Detection d:detections){
        if(d.getRect().y<threshy || abs(d.getRect().y+d.getRect().height-640) < threshy){
            needmove=true;
            cy = d.getRect().y+d.getRect().height/2;
        }
    }
    if(needmove){
        float ratio;
        if((float)roiSrc.width/(float)roiSrc.height > 1056.0/640.0) ratio=1056.0/(float)roiSrc.width;
        else ratio = 640.0/(float)roiSrc.height;
        newRoi.y = roiSrc.y  +  (int)((cy-320)/(ratio));
    }
    return needmove;
}

// 通用检测流程函数 v2
void commonDetectProcess(string &result, PVADetector &detectorValues, Mat &img, Rect &roiDate, int numberLength,
                           bool containEnglish = false, float conf = 0.2, float iouThresh=0.1){
    result="";
    // 检测流程 1
    vector<Detection> tempd;
    subCommonDetectProcess(detectorValues, img, roiDate, tempd, containEnglish,conf, iouThresh);

    if(tempd.empty()){
        cout << " first detection size = 0 , return"<< endl;
        return;
    }


    sort(tempd.begin(),tempd.end(),compareX);
    vector<Detection> temp2;
    bool ifmoved = false;
    Rect roiMoved;

    if (tempd.size()==numberLength){
        temp2=tempd;
        for (int i = 0; i < temp2.size(); ++i) {
            if (temp2[i].getClass()=="Z") result+="7";
            else result+=temp2[i].getClass();
        }
    } else {
        // 平移roi到中心
//        cout << "roi src " << roiDate.x << " " << roiDate.y << endl;
        ifmoved=roiMove(img,tempd,roiDate, roiMoved);
//        cout << "roi move "<< roiMoved.x << " " << roiMoved.y << endl;
//        cout << " move roi to center y  " << ifmoved << endl;

        // 再次检测
        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containEnglish, conf, iouThresh);


        // 去除多余
        if(temp2.size()>numberLength){
            sort(temp2.begin(),temp2.end(),compareS);
            vector<Detection> temp3;
            temp3.insert(temp3.end(),temp2.begin(),temp2.begin()+numberLength);
            sort(temp3.begin(), temp3.end(), compareX);
            for (int i = 0; i < numberLength; ++i) {
                if (temp3[i].getClass()=="Z") result+="7";
                else result+=temp3[i].getClass();
            }
            temp2.clear();
            temp2.insert(temp2.end(),temp3.begin(),temp3.end());
        }
            // 少了 就 gap 检测
        else if (temp2.size()<=numberLength) {
//            cout << "debug2" << endl;
            sort(temp2.begin(),temp2.end(),compareX);
            for (int i = 0; i < temp2.size(); ++i) {
                if (temp2[i].getClass()=="Z") result+="7";
                else result+=temp2[i].getClass();
            }
        }
    }

////         debug info
//    string saveCropImg = "/home/wurui/project/ocr-nameplates-cuizhou/item/enginemodel/crop/";
//
//    Mat subimg;
//    img(roiDate).copyTo(subimg);
//    // 1056 * 640
//    cropImg(subimg);
//    if(ifmoved){
//        subimg = img(roiMoved).clone();
//        cropImg(subimg);
//    }
//    for(int j=0;j<temp2.size();++j) {
//        Detection value = temp2[j];
//        // debug info
//        Rect number = value.getRect();
//        cout << "class: " + value.getClass() << " s :" << value.getScore()
//             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
//             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
//
////        Mat saveImg = subimg(number).clone();
////        imwrite(saveCropImg+std::to_string(count)+"_"+std::to_string(j)+".jpg", saveImg);
//
//        cv::rectangle(subimg, number, Scalar(0, 0, 255), 1);
//    }
//    cout << " >>>>> value result: "<< result<<endl;
//    cv::imshow("subimg",subimg);
//    cv::waitKey(0);
}

// 带分类网，全数字
void commonDetectProcess(Classifier &classifier, string &result, PVADetector &detectorValues, Mat &img, Rect &roiDate, int numberLength,
                           bool containEnglish = false, float conf = 0.2, float iouThresh=0.1){
    result="";
    // 检测流程 1
    vector<Detection> tempd;
    subCommonDetectProcess(detectorValues, img, roiDate, tempd, containEnglish,conf, iouThresh);

    if(tempd.empty()){
        cout << " first detection size = 0 , return"<< endl;
        return;
    }


    sort(tempd.begin(),tempd.end(),compareX);
    vector<Detection> temp2;
    bool ifmoved = false;
    Rect roiMoved;

    if (tempd.size()==numberLength){
        temp2=tempd;
        for (int i = 0; i < temp2.size(); ++i) {
            if (temp2[i].getClass()=="Z") result+="7";
            else result+=temp2[i].getClass();
        }
    } else {
        // 平移roi到中心
//        cout << "roi src " << roiDate.x << " " << roiDate.y << endl;
        ifmoved=roiMove(img,tempd,roiDate, roiMoved);
//        cout << "roi move "<< roiMoved.x << " " << roiMoved.y << endl;
//        cout << " move roi to center y  " << ifmoved << endl;

        // 再次检测
        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containEnglish, conf, iouThresh);



        // 去除多余
        if(temp2.size()>numberLength){
            sort(temp2.begin(),temp2.end(),compareS);
            vector<Detection> temp3;
            temp3.insert(temp3.end(),temp2.begin(),temp2.begin()+numberLength);
            temp2.clear();
            temp2.insert(temp2.end(),temp3.begin(),temp3.end());
        }
    }


    result = "";

    Mat subimg;
    img(roiDate).copyTo(subimg);
    // 1056 * 640
    cropImg(subimg);
    if(ifmoved){
        subimg = img(roiMoved).clone();
        cropImg(subimg);
    }

//    cv::imshow("subimg",subimg);
    sort(temp2.begin(),temp2.end(),compareX);
    for(int j=0;j<temp2.size();++j) {
        Detection value = temp2[j];
        // debug info
        Rect number = value.getRect();
        cout << "class: " + value.getClass() << " s :" << value.getScore() << endl;

        Mat cropImg = subimg(number).clone();
        resize(cropImg,cropImg,Size(224,224));
//        cv::imshow("cropImg", cropImg);
        vector<Prediction> pre = classifier.Classify(cropImg,1);
        cout << "   classify class: " << pre[0].first << " s:" << pre[0].second<<endl;
        if(value.getScore()<0.99){
            if(pre[0].second<0.85 || isNumber(pre[0].first)==false){
                result+=temp2[j].getClass();
            }
            else{
                result+=pre[0].first;
            }
        }
        else{
            result+=temp2[j].getClass();
        }


        cout << " result debug1111 "<< result << endl;
//        cv::waitKey(0);
    }


////    //     debug info
//    string saveCropImg = "/home/wurui/project/ocr-nameplates-cuizhou/item/enginemodel/crop/";
//
//    Mat subimg;
//    img(roiDate).copyTo(subimg);
//    // 1056 * 640
//    cropImg(subimg);
//    if(ifmoved){
//        subimg = img(roiMoved).clone();
//        cropImg(subimg);
//    }
//    for(int j=0;j<temp2.size();++j) {
//        Detection value = temp2[j];
//        // debug info
//        Rect number = value.getRect();
//        cout << "class: " + value.getClass() << " s :" << value.getScore()
//             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
//             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
//
//        Mat cropImg = subimg(number).clone();
//        cv::imshow("crop img", cropImg);
//        cv::waitKey(0);
//        resize(cropImg,cropImg,Size(224,224));
//        vector<Prediction> pre = classifier.Classify(cropImg,1);
//        cout << " classify class: " << pre[0].first << " s:" << pre[0].second<<endl;
////        cv::rectangle(subimg, number, Scalar(0, 0, 255), 1);
//    }
//    cout << " >>>>> value result: "<< result<<endl;
//    cv::imshow("subimg",subimg);
//    cv::waitKey(0);
}

// 整车型号
void commonDetectProcessForVehicleModel(string &result, PVADetector &detectorValues, Mat &img, Rect &roiDate, int numberLength,
                           bool containEnglish = false, float conf = 0.2, float iouThresh=0.1){
    result="";
    // 检测流程 1
    vector<Detection> tempd;
    subCommonDetectProcess(detectorValues, img, roiDate, tempd, containEnglish,conf, iouThresh);

    if(tempd.empty()){
        cout << " first detection size = 0 , return"<< endl;
        return;
    }

//    // gap detect
//    gapDetect(detectorValues,tempd, roiDate, img);
    // iou
    cout << "size before iou "<< tempd.size() <<endl;
    iouOverlap(tempd, 0.5);
    cout << "size after iou "<< tempd.size() <<endl;

    sort(tempd.begin(),tempd.end(),compareX);
    vector<Detection> temp2;
    bool ifmoved = false;
    Rect roiMoved;

    if (tempd.size()==numberLength){
        temp2=tempd;
        for (int i = 0; i < temp2.size(); ++i) {
            if (temp2[i].getClass()=="Z") result+="7";
            else result+=temp2[i].getClass();
        }
    } else {
        // 平移roi到中心
        cout << "roi src " << roiDate.x << " " << roiDate.y << endl;
        ifmoved=roiMove(img,tempd,roiDate, roiMoved);
        cout << "roi move "<< roiMoved.x << " " << roiMoved.y << endl;
//        cout << " move roi to center y  " << ifmoved << endl;

        // 再次检测
        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containEnglish, conf, iouThresh);

        cout << " debug1 " << endl;



        // 去除多余
        if(temp2.size()>numberLength){
            sort(temp2.begin(),temp2.end(),compareS);
            vector<Detection> temp3;
            temp3.insert(temp3.end(),temp2.begin(),temp2.begin()+numberLength);
            sort(temp3.begin(), temp3.end(), compareX);
            for (int i = 0; i < numberLength; ++i) {
                if (temp3[i].getClass()=="Z") result+="7";
                else result+=temp3[i].getClass();
            }
            temp2.clear();
            temp2.insert(temp2.end(),temp3.begin(),temp3.end());
        }
            // 少了 就 gap 检测
        else if (temp2.size()<=numberLength) {
//            cout << "debug2" << endl;
            sort(temp2.begin(),temp2.end(),compareX);
            for (int i = 0; i < temp2.size(); ++i) {
                if (temp2[i].getClass()=="Z") result+="7";
                else result+=temp2[i].getClass();
            }
        }
    }

    //     debug info
//    string saveCropImg = "/home/wurui/project/ocr-nameplates-cuizhou/item/enginemodel/crop/";
//
//    Mat subimg;
//    img(roiDate).copyTo(subimg);
//    // 1056 * 640
//    cropImg(subimg);
//    if(ifmoved){
//        subimg = img(roiMoved).clone();
//        cropImg(subimg);
//    }
//    for(int j=0;j<temp2.size();++j) {
//        Detection value = temp2[j];
//        // debug info
//        Rect number = value.getRect();
//        cout << "class: " + value.getClass() << " s :" << value.getScore()
//             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
//             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
//
////        Mat saveImg = subimg(number).clone();
////        imwrite(saveCropImg+std::to_string(count)+"_"+std::to_string(j)+".jpg", saveImg);
//
//        cv::rectangle(subimg, number, Scalar(0, 0, 255), 1);
//    }
//    cout << " >>>>> value result: "<< result<<endl;
////    cv::imshow("subimg",subimg);
////    cv::waitKey(0);
}

// 生产日期
string DateOfManufacture(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect){
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width+5;
    y = keyRect.y-2;
    w = max((int)(keyRect.width*1.5),150);
    h = (int)(keyRect.height*1.2);
    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 6;
    commonDetectProcess(result, detectorValues, img, roiDate, valueNumberLen);
    return result;
}

// 最大允许总质量
string MaxMassAllowed(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect){
    string result="";

    Mat image = img.clone();
    const int numberLength = 4;

    // --------stage1
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width;
    y = keyRect.y+3;
    w = max((int)(keyRect.width*0.1),95);
    h = (int)max(keyRect.height*1,45);
    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 4;
    commonDetectProcess(result, detectorValues, img, roiDate, valueNumberLen);

    return result;
}

//发动机最大功率
string MaxNetPowerOfEngine(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect){
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width;
    y = keyRect.y+10;
    w = max((int)(keyRect.width*0.1),90);
    h = (int)max(keyRect.height*1,40);
    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
//    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 3;

    commonDetectProcess(result, detectorValues, img, roiDate, valueNumberLen);

    if (result==""){
        cout << " fisrt none,roi move detect again" << endl;
        x = keyRect.x+keyRect.width;
        y = keyRect.y-5;
        w = max((int)(keyRect.width*0.1),90);
        h = (int)max(keyRect.height*1,40);
        cout << x <<" x "<<y<<" y "<<endl;
        cv::Rect roiDate1(x,y,max(0,w),max(0,h));
        cv::rectangle(image, roiDate1, Scalar(255, 0, 0), 1);
//        cv::imshow("roi2", image);
//        cv::waitKey(0);
        commonDetectProcess(result, detectorValues, img, roiDate1, valueNumberLen);

    }

    for (int i = 0; i < result.size(); ++i) {
        if (result.substr(i,1)=="5" ||result.substr(i,1)=="3"||result.substr(i,1)=="6"||result.substr(i,1)=="2"||result.substr(i,1)=="0" ) {
            result = "206";
        } else if (result.substr(i,1)=="1" ||result.substr(i,1)=="4"||result.substr(i,1)=="7"){
            result = "147";
        }
    }

    cout << result << " in func"<<endl;


    cout << "result ===== " << result<<endl;

    return result;
}

// 涂料
string painMatch(string str1, string str2){
    string names[11] = {"414", "361", "217", "248", "092", "093", "035", "318", "408", "409","620"};
    if (str1=="0" && str2=="2")
        return "092";
    if (str1=="4" && str2=="4")
        return "414";
    string result = "";
    int flag = -1;
    for (int i = 0; i < 11; ++i) {
        string s1,s2,s3;
        s1 = names[i].substr(0,1);
        s2 = names[i].substr(1,1);
        s3 = names[i].substr(2,1);
//        cout << " match <<<<" << endl;
//        cout << str1 << " " << str2 << endl;
        if((str1==s1 && str2==s2) || (str1==s2 && str2==s3)) {
//            cout << "flag =======" <<  names[flag]<<endl;
            flag=i;
        }
    }
    if(flag>-1){
        result = names[flag];
    }

    return result;
}

string Paint(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect){
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width;
    y = keyRect.y;
    w = max((int)(keyRect.width*0.1),45);
    h = (int)max(keyRect.height*1,30);
    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 3;
    commonDetectProcess(result, detectorValues, img, roiDate, valueNumberLen, 0.05);



    cout << "result size " << result.size() << endl;
    if(result.size()==2){
        string s1 = result.substr(0,1);
        string s2 = result.substr(1,1);
        result = painMatch(s1, s2);
    }
    else if (result.size()==3){
        string s1 = result.substr(0,1);
        string s2 = result.substr(1,1);
        string resultMatch = painMatch(s1, s2);
        cout << "resultMatch " << resultMatch<<endl;
        if (resultMatch==""){
             s1 = result.substr(1,1);
            cout << "debug "<< endl;
             s2 = result.substr(2,1);
            resultMatch = painMatch(s1, s2);
            cout << "resultMatch " << resultMatch<<endl;
        }
        result = resultMatch;
        cout << "resultMatch " << resultMatch<<endl;
    }

    cout << " >>> match Paint value =" << result << endl;
//    Mat subimg;
//    img(roiDate).copyTo(subimg);
//
//    // 填充
//    Mat detectImg(subimg.rows,subimg.cols*1,CV_8UC3,Scalar(0,0,0));
//    Rect rcDetect(Point(max((int)(detectImg.cols /2 - subimg.cols/2),0), 0), subimg.size());
//    subimg.copyTo(detectImg(rcDetect));
//
//    detectorValues.setThresh(0.2,0.1);
//    vector<Detection> valueDate = detectorValues.detect(detectImg);
//
//    std::sort(valueDate.begin(),valueDate.end(),compareX);
//
//    // iou 去重叠
//    eliminateOverlaps(valueDate);
//
//    vector<Detection> temp;
//    for(Detection d:valueDate){
//        string cls = d.getClass();
//        if (isNumber(cls)) temp.push_back(d);
//    }
//    cout << "raw detection num: "<< temp.size()<<endl;
//    sort(temp.begin(), temp.end(), compareX);
//
//    vector<Detection> temp2;
//    if (temp.size()>3){
//        sort(temp.begin(),temp.end(),compareS);
//        temp2.insert(temp2.end(), temp.begin(), temp.begin()+3);
//        sort(temp2.begin(), temp2.end(), compareX);
//        for (int i = 0; i < temp2.size(); ++i) {
//            result+=temp2[i].getClass();
//        }
//    } else if(temp.size()==2){
//        temp2 = temp;
//        string s1 = temp[0].getClass();
//        string s2 = temp[1].getClass();
//        result = painMatch(s1, s2);
//
//        cout << " match result:" << result << endl;
////        if(temp[0].getClass()=="4" && temp[1].getClass()=="1") result="414";
//        if(temp[0].getClass()=="4" && temp[1].getClass()=="4") result="414";
////        if(temp[0].getClass()=="1" && temp[1].getClass()=="4") result="414";
////        if(temp[0].getClass()=="1" && temp[1].getClass()=="7") result="217";
////        if(temp[0].getClass()=="2" && temp[1].getClass()=="1") result="217";
////        if(temp[0].getClass()=="0" && temp[1].getClass()=="9") result="092";
////        if(temp[0].getClass()=="9" && temp[1].getClass()=="2") result="092";
////        if(temp[0].getClass()=="0" && temp[1].getClass()=="3") result="035";
//    }
//    else {
//        temp2=temp;
//        for (int i = 0; i < temp2.size(); ++i) {
//            result+=temp2[i].getClass();
//        }
//    }
//
////     debug info
//    for(int j=0;j<temp2.size();++j) {
//        Detection value = temp2[j];
//        // debug info
//        Rect number = value.getRect();
//        cout << "class: " + value.getClass() << " s :" << value.getScore()
//             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
//             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
////        number.x += roiDate.x;
////        number.y += roiDate.y;
//        cv::rectangle(detectImg, number, Scalar(0, 0, 255), 1);
//    }
//
//    cout << "DateOfManufacture result: "<< result<<endl;
//    cv::imshow("detectimg",detectImg);
//    int key = cv::waitKey(0);
//    if (key>0) cout << endl;
    return result;
}

// 发动机型号 55273835
string EngineModel(Classifier &classifier, Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect)
{
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width+5;
    y = keyRect.y;
    w = max((int)(keyRect.width*0.1),150);
    h = (int)max(keyRect.height*1,40);
//    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 8;
    commonDetectProcess(classifier, result, detectorValues, img, roiDate, valueNumberLen);
    cout << " result << " << result << endl;

//    Mat subimg;
//    img(roiDate).copyTo(subimg);
//
//    // 填充
//    Mat detectImg(subimg.rows,subimg.cols*1.5,CV_8UC3,Scalar(0,0,0));
//    Rect rcDetect(Point(max((int)(detectImg.cols /2 - subimg.cols/2),0), 0), subimg.size());
//    subimg.copyTo(detectImg(rcDetect));
//
//    detectorValues.setThresh(0.2,0.1);
//    vector<Detection> valueDate = detectorValues.detect(detectImg);
//
//    std::sort(valueDate.begin(),valueDate.end(),compareX);
//
//    // iou 去重叠
//    eliminateOverlaps(valueDate);
//
//    // 去除不在一条线上的
//    vector<Detection> tempd;
//    deleteBoxNotLine(valueDate,tempd);
//
//    vector<Detection> temp;
//    for(Detection d:tempd){
//        string cls = d.getClass();
//        if (isNumber(cls)) temp.push_back(d);
//    }
//
//    // gap
//    Mat image2 = img.clone();
//    gapDetect(detectorValues,temp,roiDate,image2);
//    sort(temp.begin(),temp.end(),compareX);
//
//    vector<Detection> temp2;
//    if (temp.size()>8){
//        sort(temp.begin(),temp.end(),compareS);
//        temp2.insert(temp2.end(), temp.begin(), temp.begin()+8);
//        sort(temp2.begin(), temp2.end(), compareX);
//        for (int i = 0; i < temp2.size(); ++i) {
//            result+=temp2[i].getClass();
//        }
//    } else if(temp.size()<8){
//        temp2=temp;
//        for (int i = 0; i < temp2.size(); ++i) {
//            result+=temp2[i].getClass();
//        }
//    } else {
//        temp2=temp;
//        for (int i = 0; i < temp2.size(); ++i) {
//            result+=temp2[i].getClass();
//        }
//    }
//////     debug info
////    for(int j=0;j<temp2.size();++j) {
////        Detection value = temp2[j];
////        // debug info
////        Rect number = value.getRect();
////        cout << "class: " + value.getClass() << " s :" << value.getScore()
////             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
////             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
//////        number.x += roiDate.x;
//////        number.y += roiDate.y;
//////        cv::rectangle(image, number, Scalar(0, 0, 255), 1);
////        cv::rectangle(detectImg, number, Scalar(0, 0, 255), 1);
////    }
////
////    cout << "DateOfManufacture result: "<< result<<endl;
////    cv::imshow("img",detectImg);
////    int key = cv::waitKey(0);
////    if (key>0) cout << endl;
    return result;
}

// 乘坐人数
string NumPassengers(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect)
{
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width+5;
    y = keyRect.y;
    w = max((int)(keyRect.width*0.1),45);
    h = (int)max(keyRect.height*1,40);
//    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

//    int valueNumberLen = 1;
//    commonDetectProcessV2(result, detectorValues, img, roiDate, valueNumberLen);
//    return result;

    Mat subimg;
    img(roiDate).copyTo(subimg);

    detectorValues.setThresh(0.2,0.1);
    vector<Detection> valueDate = detectorValues.detect(subimg);

    cout << "valueDate size: " << valueDate.size()<<endl;

    std::sort(valueDate.begin(),valueDate.end(),compareX);

    // iou 去重叠
    eliminateOverlaps(valueDate);

    vector<Detection> temp;
    for(Detection d:valueDate){
        string cls = d.getClass();
        if (isNumber(cls)) temp.push_back(d);
    }

    cout << "isNumber size: " << temp.size()<<endl;

    vector<Detection> temp2;
    if (temp.size()>1){
        sort(temp.begin(),temp.end(),compareS);
        temp2.insert(temp2.end(), temp.begin(), temp.begin()+1);
        sort(temp2.begin(), temp2.end(), compareX);
        for (int i = 0; i < temp2.size(); ++i) {
            result+=temp2[i].getClass();
        }
    }  else {
        temp2=temp;
        for (int i = 0; i < temp2.size(); ++i) {
            result+=temp2[i].getClass();
        }
    }

    cout << "temp2 size: " << temp2.size()<<endl;
    return result;

//    // debug info
//    for(int j=0;j<temp2.size();++j) {
//        Detection value = temp2[j];
//        // debug info
//        Rect number = value.getRect();
//        cout << "class: " + value.getClass() << " s :" << value.getScore()
//             << " rect: " << std::to_string(number.x) + " " + std::to_string(number.y)
//             << " " << std::to_string(number.width) << " " <<std::to_string(number.height)<< endl;
//        number.x += roiDate.x;
//        number.y += roiDate.y;
//        cv::rectangle(image, number, Scalar(0, 0, 255), 1);
//    }
//
//    cout << "DateOfManufacture result: "<< result<<endl;
//    cv::imshow("img",image);
//    int key = cv::waitKey(0);
//    if (key>0) cout << endl;
}

// 整车型号  带英文
string VehicleModel(Classifier &classifier, Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect)
{
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = min(keyRect.x+keyRect.width+5,image.cols-1);
    y = keyRect.y;
    w = max(min((int)(image.cols-1-x),120),0);
    h = (int)max(keyRect.height*1,40);
//    cout << x <<" x "<<y<<" y "<<endl;
//    cout << image.cols << " * "<< image.rows << endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 8;
    bool containEnglishChar = true;
    commonDetectProcessForVehicleModel(result, detectorValues, img, roiDate, valueNumberLen, containEnglishChar);
    return result;
}

// 发动机排量
string EngineDisplacement(Mat &img, PVADetector &detectorKeys, PVADetector &detectorValues, Rect &keyRect)
{
    string result="";
    Mat image = img.clone();
    cv::rectangle(image, keyRect, Scalar(255, 0, 0), 1);
    int x,y,w,h;
    x = keyRect.x+keyRect.width+5;
    y = keyRect.y;//keyRect.y-5;
    w = max((int)(keyRect.width*0.1),100);;//max((int)(keyRect.width*0.1),120);
    h = (int)max(keyRect.height*1,35);//(int)max(keyRect.height*1,40);
//    cout << x <<" x "<<y<<" y "<<endl;
    cv::Rect roiDate(x,y,max(0,w),max(0,h));
    cv::rectangle(image, roiDate, Scalar(255, 0, 0), 1);

    int valueNumberLen = 4;
    commonDetectProcess(result, detectorValues, img, roiDate, valueNumberLen);
    return result;
}

// debug show result
void drawResult(Mat &image, Rect &roi, string &result){
    cv::putText(image,result,Point(roi.x,roi.y+8),0,0.8,Scalar(0,0,255),2);
}

// ---------------- wurui end --------------------------//

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace cv;
    namespace fs = boost::filesystem;

    string pathInputDir = "/home/wurui/project/ocr-nameplates-cuizhou/data-alfaromeo/" ; //data-alfaromeo/" ;//item2/maxmass/false-src/";
    string tempdir = "/home/wurui/project/ocr-nameplates-cuizhou/";

    string pathModelKeys = "/home/wurui/project/ocr-nameplates-cuizhou/models/table_header_model/car_brand_iter_100000.caffemodel";
    string pathPtKeys = "/home/wurui/project/ocr-nameplates-cuizhou/models/table_header_model/test.prototxt";
    vector<string> classesKeys = readClassNames(
            "/home/wurui/project/ocr-nameplates-cuizhou/models/table_header_model/classes_name.txt");

    // old model
    string pathModelValuesOld = "/home/wurui/project/ocr-nameplates-cuizhou/models/single_char_model/alfa_engnum_char_iter_100000.caffemodel";
    string pathPtValuesOld = "/home/wurui/project/ocr-nameplates-cuizhou/models/single_char_model/test.prototxt";
    vector<string> classesCharsOld = readClassNames(
            "/home/wurui/project/ocr-nameplates-cuizhou/models/single_char_model/classes_name.txt");

    // new model
    string pathModelValues = "/home/wurui/project/ocr-nameplates-cuizhou/models/pvashape_num_engchar/alfa_char_shape_pva_iter_100000.caffemodel";
    string pathPtValues = "/home/wurui/project/ocr-nameplates-cuizhou/models/pvashape_num_engchar/test.prototxt";
    vector<string> classesChars = readClassNames(
            "/home/wurui/project/ocr-nameplates-cuizhou/models/pvashape_num_engchar/classes_name.txt");

    // googlenet
    string model_file = "/home/wurui/project/ocr-nameplates-cuizhou/models/googlenet/deploy.prototxt";
    string trained_file = "/home/wurui/project/ocr-nameplates-cuizhou/models/googlenet/model_googlenet_iter_38942.caffemodel";
    string mean_file = "/home/wurui/project/ocr-nameplates-cuizhou/models/googlenet/mean.binaryproto";
    string label_file = "/home/wurui/project/ocr-nameplates-cuizhou/models/googlenet/classname.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    // std::vector<Prediction> predictions = classifier.Classify(img); Prediction = std::pair<string,float>


    PVADetector detectorKeys;
    detectorKeys.init(pathPtKeys, pathModelKeys, classesKeys);
    detectorKeys.setThresh(0.5, 0.1);
    detectorKeys.setComputeMode("gpu", 0);

    // old Model
    PVADetector detectorValuesOld;
    detectorValuesOld.init(pathPtValuesOld, pathModelValuesOld, classesCharsOld);
    detectorValuesOld.setThresh(0.05, 0.3);
    detectorValuesOld.setComputeMode("gpu", 0);

    PVADetector detectorValues;
    detectorValues.init(pathPtValues, pathModelValues, classesChars);
    detectorValues.setThresh(0.05, 0.3);
    detectorValues.setComputeMode("gpu", 0);


    int countAll = 0, countCorrect = 0, countShorter = 0, countLonger= 0, countWrong = 0;

    int count=0;
    vector<string> imagesNames;
    getFileNames(pathInputDir,imagesNames);
    for (string pathImg:imagesNames){
        cout << "img: "<< pathImg<<endl;

        bool isRight = true;
        Mat img = imread(pathImg);
        Mat image = img.clone();
        if (img.empty()) continue;

        vector<Detection> keyDets = detectorKeys.detect(img);
        for (auto const& keyDet: keyDets) {
            string key = keyDet.getClass();
            Rect keyRect = keyDet.getRect();
            string value;

            if (key == "Manufacturer") {

            }
            else if (key == "Brand") {

            }
            else if (key == "MaxMassAllowed") {
                try {
                    cout << "MaxMassAllowed" << endl;
                    string result =  MaxMassAllowed(img, detectorKeys, detectorValues, keyRect);
                    if (result != "2150" && result!="2175") isRight = false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error MaxMassAllowed" << endl;
                }
            }
         else if (key == "MaxNetPowerOfEngine") {
                try {
                    cout << "MaxNetPowerOfEngine" << endl;
                    string result = MaxNetPowerOfEngine(img, detectorKeys, detectorValues, keyRect);
                    cout << "result >>>>>>>>>" << result << endl;
                    if(result !="147" && result!="206") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error MaxNetPowerOfEngine" << endl;
                }
            }
        else if (key == "Country") {

            }
            else if (key == "Factory") {

            }
            else if (key == "EngineModel") { // 发动机型号
                try {
                    cout << "EngineModel" << endl;
                    string result = EngineModel(classifier, img, detectorKeys, detectorValues, keyRect);
                    cout << " result >>>>>> "<<result << endl;
                    cout << (result=="55273835") << endl;
                    if(result != "55273835") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error EngineModel" << endl;
                }
            }
            else if (key == "NumPassengers") { // 乘坐人数
                try {
                    cout << "NumPassengers" << endl;
                    string result = NumPassengers(img, detectorKeys, detectorValuesOld, keyRect);
                    if(result != "5") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error NumPassengers" << endl;
                }
            }

            else if (key == "VehicleId") {

            }
//
            else if (key == "VehicleModel") { // 整车型号
                try {
                    cout << "VehicleModel" << endl;
                    string result = VehicleModel(classifier, img, detectorKeys, detectorValues, keyRect);
                    if(result!="AR952CA2" && result!="AR952BA2") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error VehicleModel" << endl;
                }
            }
            else if (key == "EngineDisplacement") { // 发动机排量
                try {
                    cout << "EngineDisplacement" << endl;
                    string result = EngineDisplacement(img, detectorKeys, detectorValues, keyRect);
                    if(result != "1995") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error EngineDisplacement" << endl;
                }
            }
            else if (key =="DateOfManufacture") { // 生产日期
                try {
                    cout << "DateOfManufacture" << endl;
                    string result = DateOfManufacture(img, detectorKeys, detectorValues, keyRect);
                    if(result != "201703" && result != "201704" && result != "201701") isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error DateOfManufacture" << endl;
                }
            }
            else if (key == "Paint") { // 涂料
                try {
                    cout << "Paint" << endl;
                    string result = Paint(img, detectorKeys, detectorValues, keyRect);
                    if(result.size() != 3) isRight=false;
                    drawResult(image,keyRect,result);
                } catch (int a){
                    cout << "error  Paint" << endl;
                }
            }
        }

        count+=1;
        cout << " total count : " << count << endl;
        cout << " flag ====== " << isRight << endl;
//        cv::imshow("img",image);
//        int key = cv::waitKey(0);

        string filename = tempdir+"item2/img.txt";
        ofstream wimg;
        wimg.open(filename,std::ios::out | std::ios::app);
        if(isRight){
            cv::imwrite(tempdir+"item2/true/"+std::to_string(count)+".jpg",image);
        } else{
            wimg << pathImg << endl;
            cv::imwrite(tempdir+"item2/false/"+std::to_string(count)+".jpg",image);
            cv::imwrite(tempdir+"item2/false-src/"+std::to_string(count)+".jpg",img);
        }
        wimg.close();

//        cv::imwrite(tempdir+"temp/"+std::to_string(count)+".jpg",image);


        cout << countCorrect << " out of " << countAll << " (" << 100 * float(countCorrect) / countAll << "%) correct." << "...";
        cout << countWrong << " out of " << countAll << " (" << 100 * float(countWrong) / countAll << "%) wrong." << "...";
        cout << countShorter << " out of " << countAll << " (" << 100 * float(countShorter) / countAll << "%) shorter." << "...";
        cout << countLonger << " out of " << countAll << " (" << 100 * float(countLonger) / countAll << "%) longer." << endl;
    }

//    waitKey(0);
    return 0;
}
