#include "Common.h"

//----------------------------------------------------
/**
\brief   copy image
\param   in                 [in]  input image
\param   out                [out] output image
\return  void
*/
//-----------------------------------------------------
void copy(Image &in, Image &out)
{
    out.height = in.height;
    out.width = in.width;
    out.channels = in.channels;
}

//----------------------------------------------------
/**
\brief   Mat to Image
\param   in                 [in]  input image
\param   out                [out] output image
\return  void
*/
//-----------------------------------------------------
void Mat2Image(cv::Mat &in, Image &out)
{
    out.height = in.rows;
    out.width = in.cols;
    out.channels = in.channels();


    for(int i = 0; i != in.rows; ++i)
    {
        unsigned char *data = in.ptr<unsigned char>(i);
        for(int j = 0; j != in.cols * in.channels(); ++j)
        {
            if(in.channels() == 3)
            {
                int num = j % in.channels();
                int size = in.rows * in.cols;
                out.data[i * in.cols + j / 3 + size * num] = data[j];
            }
            else
            {
                out.data[i * in.cols + j] = data[j];
            }
        }
    }
}

//----------------------------------------------------
/**
\brief   Image to Mat
\param   in                 [in]  input image
\param   out                [out] output image
\return  void
*/
//-----------------------------------------------------
void Image2Mat(Image &in, cv::Mat &out)
{
    if(in.channels == 1)
    {
        out.create(in.height, in.width, CV_8U);
        for(int i = 0; i != out.rows; ++i)
        {
            unsigned char *data = out.ptr<unsigned char>(i);
            for(int j = 0; j != out.cols * out.channels(); ++j)
            {
                data[j] = in.data[i * in.width + j];
            }
        }
    }
    if(in.channels == 3)
    {
        out.create(in.height, in.width, CV_8UC3);
        int size = in.height * in.width;
        for(int i = 0; i != out.rows; ++i)
        {
            unsigned char *data = out.ptr<unsigned char>(i);
            for(int j = 0; j != out.cols * out.channels(); ++j)
            {
                int num = j % in.channels;
                data[j] = in.data[i * in.width + j / 3 + size * num];
            }
        }
    }
}

//----------------------------------------------------
/**
\brief   BGR to GRAY
\param   in                 [in]  input image
\param   out                [out] output image
\return  void
*/
//-----------------------------------------------------
void BGR2GRAY(Image &in, Image &out)
{
    out.height = in.height;
    out.width = in.width;
    out.channels = 1;

    int length = in.height * in.width;
    int offset = in.height * in.width;
    for(int i = 0; i != length; ++i)
    {
        int b = in.data[i];
        int g = in.data[i + offset];
        int r = in.data[i + 2 * offset];
        int gray = (r * 38 +  g * 75 +  b * 15) >> 7;
        out.data[i] = CLIP8(gray);
    }
}

//----------------------------------------------------
/**
\brief   display Image
\param   window            [in]  window name
\param   display           [in]  image to be displayed
\return  void
*/
//-----------------------------------------------------
void ShowImage(std::string window, Image &display)
{
    cv::Mat mdisplay;
    Image2Mat(display, mdisplay);
    cv::imshow(window, mdisplay);
}
