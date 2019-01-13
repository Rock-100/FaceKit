#ifndef __COMMON__
#define __COMMON__

#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define PIXEL_MAX 255
#define PIXEL_MIN 0

#define  CLAMP(x, l, u)   ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))

#define  CLIP8(a)         (((a) & 0xFFFFFF00) ? (((a) < 0) ? 0 : 255 ) : (a))

///data structure for image
struct Image
{
    unsigned char *data;
    int height;
    int width;
    int channels;
};

///copy image
void copy(Image &in, Image &out);

///Mat to Image
void Mat2Image(cv::Mat &in, Image &out);

///Image to Mat
void Image2Mat(Image &in, cv::Mat &out);

///display Image
void ShowImage(std::string window, Image &display);

void BGR2GRAY(Image &in, Image &out);

#endif
