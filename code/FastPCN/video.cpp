#include "FastPCN.h"

int main()
{
    FastPCN detector("FastPCN.caffemodel");
    detector.SetMinFaceSize(48);
    detector.SetScoreThresh(0.95);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetVideoSmooth(true);

    cv::VideoCapture capture(0);
    cv::Mat img;
    while (1)
    {
        capture >> img;
        std::vector<Window> faces = detector.DetectFace(img);

        for (int i = 0; i < faces.size(); i++)
        {
            DrawFace(img, faces[i]);
        }
        cv::imshow("PCN", img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
