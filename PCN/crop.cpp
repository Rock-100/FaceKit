#include "PCN.h"

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

int main()
{
    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel",
                 "model/PCN-Tracking.prototxt");
    /// detection
    detector.SetMinFaceSize(45);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetDetectionThresh(0.37, 0.43, 0.97);
    /// tracking
    detector.SetTrackingPeriod(30);
    detector.SetTrackingThresh(0.95);
    detector.SetVideoSmooth(false);

    for (int i = 1; i <= 26; i++)
    {
        cv::Mat img =
            cv::imread("imgs/" + std::to_string(i) + ".jpg");
        cv::TickMeter tm;
        tm.reset();
        tm.start();
        std::vector<Window> faces = detector.Detect(img);
        tm.stop();
        std::cout << "Image: " << i << std::endl;
        std::cout << "Time Cost: "<<
                  tm.getTimeMilli() << " ms" << std::endl;
        cv::Mat faceImg;
        for (int j = 0; j < faces.size(); j++)
        {
            cv::Mat tmpFaceImg = CropFace(img, faces[j], 200);
            faceImg = MergeImgs(faceImg, tmpFaceImg);
        }
        cv::imshow("Faces", faceImg);
        for (int j = 0; j < faces.size(); j++)
        {
            DrawFace(img, faces[j]);
        }
        cv::imshow("PCN", img);
        cv::waitKey();
    }

    cv::destroyAllWindows();

    return 0;
}
