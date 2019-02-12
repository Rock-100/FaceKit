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

struct Window
{
    int x, y, width;
    float angle, score;
    int id;
    Window(int x_, int y_, int w_, float a_, float s_, int id_)
        : x(x_), y(y_), width(w_), angle(a_), score(s_), id(id_)
    {}
};

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
    void SetIOUThresh(float high_thresh, float low_thresh);
    std::vector<Window> DetectTrack(cv::Mat img);

private:
    void* impl_;
};

extern "C"{
	cv::Point RotatePoint(int x, int y, float centerX, float centerY, float angle);
	void DrawLine(cv::Mat img, std::vector<cv::Point> pointList);
	void DrawFace(cv::Mat img, Window face);
	cv::Mat CropFace(cv::Mat img, Window face, int cropSize);
	void *init_detector(int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
			float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
			float tracking_thresh, int do_smooth, float iou_high_thresh, float iou_low_thresh);
	Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, size_t *lwin);
	void free_detector(void *pcn);
}

#endif

