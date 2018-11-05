#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#define M_PI  3.14159265358979323846
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#define EPS  1e-5

struct Window
{
    int x, y, width;
    float angle, score;
    Window(int x_, int y_, int w_, float a_, float s_)
        : x(x_), y(y_), width(w_), angle(a_), score(s_)
    {}
};

cv::Point RotatePoint(int x, int y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    int rx = int(centerX + x * std::cos(theta) - y * std::sin(theta));
    int ry = int(centerY + x * std::sin(theta) + y * std::cos(theta));
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int thick = 2;
    CvScalar cyan = CV_RGB(0, 255, 255);
    CvScalar blue = CV_RGB(0, 0, 255);
    cv::line(img, pointList[0], pointList[1], cyan, thick);
    cv::line(img, pointList[1], pointList[2], cyan, thick);
    cv::line(img, pointList[2], pointList[3], cyan, thick);
    cv::line(img, pointList[3], pointList[0], blue, thick);
}

void DrawFace(cv::Mat img, Window face)
{
    int x1 = face.x;
    int y1 = face.y;
    int x2 = face.width + face.x - 1;
    int y2 = face.width + face.y - 1;
    int centerX = (x1 + x2) / 2;
    int centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));
    DrawLine(img, pointList);
}

cv::Mat CropFace(cv::Mat img, Window face, int cropSize)
{
    int x1 = face.x;
    int y1 = face.y;
    int x2 = face.width + face.x - 1;
    int y2 = face.width + face.y - 1;
    int centerX = (x1 + x2) / 2;
    int centerY = (y1 + y2) / 2;
    cv::Point2f srcTriangle[3];
    cv::Point2f dstTriangle[3];
    srcTriangle[0] = RotatePoint(x1, y1, centerX, centerY, face.angle);
    srcTriangle[1] = RotatePoint(x1, y2, centerX, centerY, face.angle);
    srcTriangle[2] = RotatePoint(x2, y2, centerX, centerY, face.angle);
    dstTriangle[0] = cv::Point(0, 0);
    dstTriangle[1] = cv::Point(0, cropSize - 1);
    dstTriangle[2] = cv::Point(cropSize - 1, cropSize - 1);
    cv::Mat rotMat = cv::getAffineTransform(srcTriangle, dstTriangle);
    cv::Mat ret;
    cv::warpAffine(img, ret, rotMat, cv::Size(cropSize, cropSize));
    return ret;
}

cv::Mat MergeImgs(cv::Mat A, cv::Mat B)
{
    int totalCols = A.cols + B.cols;
    int totalRows = std::max(A.rows, B.rows);
    cv::Mat ret(totalRows, totalCols, CV_8UC3);
    cv::Mat subMat = ret.colRange(0, A.cols);
    A.copyTo(subMat);
    subMat = ret.colRange(A.cols, totalCols);
    B.copyTo(subMat);
    return ret;
}

class PCN
{
public:
    PCN(std::string model, std::string net1, std::string net2, std::string net3);
    void SetMinFaceSize(int minFace);
    void SetScoreThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    void SetVideoSmooth(bool smooth);
    std::vector<Window> DetectFace(cv::Mat img);

private:
    void* impl_;
};

#endif
