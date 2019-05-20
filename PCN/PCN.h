#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#define M_PI  3.14159265358979323846
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#define EPS  1e-5

#define CYAN CV_RGB(0, 255, 255)
#define BLUE CV_RGB(0, 0, 255)
#define GREEN CV_RGB(0, 255, 0)
#define RED CV_RGB(255, 0, 0)
#define PURPLE CV_RGB(139, 0, 255)

struct Window
{
    int x, y, width, angle;
    float score;
    std::vector<cv::Point> points14;
    Window(int x_, int y_, int w_, int a_, float s_, std::vector<cv::Point> p14_)
        : x(x_), y(y_), width(w_), angle(a_), score(s_), points14(p14_)
    {}
};

cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    float rx = centerX + x * std::cos(theta) - y * std::sin(theta);
    float ry = centerY + x * std::sin(theta) + y * std::cos(theta);
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int width = 2;
    cv::line(img, pointList[0], pointList[1], CYAN, width);
    cv::line(img, pointList[1], pointList[2], CYAN, width);
    cv::line(img, pointList[2], pointList[3], CYAN, width);
    cv::line(img, pointList[3], pointList[0], BLUE, width);
}

void DrawFace(cv::Mat img, Window face)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));
    DrawLine(img, pointList);
}

void DrawPoints(cv::Mat img, Window face)
{
    int width = 2;
    if (face.points14.size() == 14)
    {
        for (int i = 1; i <= 8; i++)
        {
            cv::line(img, face.points14[i - 1], face.points14[i], BLUE, width);
        }
        for (int i = 0; i < face.points14.size(); i++)
        {
            if (i <= 8)
                cv::circle(img, face.points14[i], width, CYAN, -1);
            else if (i <= 9)
                cv::circle(img, face.points14[i], width, GREEN, -1);
            else if (i <= 11)
                cv::circle(img, face.points14[i], width, PURPLE, -1);
            else
                cv::circle(img, face.points14[i], width, RED, -1);
        }
    }
}

cv::Mat CropFace(cv::Mat img, Window face, int cropSize)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
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

class PCN
{
public:
    PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
        std::string modelTrack, std::string netTrack);
    /// detection
    void SetMinFaceSize(int minFace);
    void SetDetectionThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    std::vector<Window> Detect(cv::Mat img);
    /// tracking
    void SetTrackingPeriod(int period);
    void SetTrackingThresh(float thresh);
    void SetVideoSmooth(bool smooth);
    std::vector<Window> DetectTrack(cv::Mat img);

private:
    void* impl_;
};

#endif
