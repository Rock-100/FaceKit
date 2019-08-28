#include "PCN_API.h"

int main()
{
    const char *detection_model_path = "./model/PCN.caffemodel";
    const char *pcn1_proto = "./model/PCN-1.prototxt";
    const char *pcn2_proto = "./model/PCN-2.prototxt";
    const char *pcn3_proto = "./model/PCN-3.prototxt";
    const char *tracking_model_path = "./model/PCN-Tracking.caffemodel";
    const char *tracking_proto = "./model/PCN-Tracking.prototxt";

    PCN* detector = (PCN*) init_detector(detection_model_path, pcn1_proto, pcn2_proto, pcn3_proto,
                                         tracking_model_path, tracking_proto,
                                         15, 1.45, 0.5, 0.5, 0.98, 30, 0.9, 0);
    for (int i = 0; i <= 26; i++)
    {
        cv::Mat img = cv::imread("imgs/" + std::to_string(i) + ".jpg");
        cv::TickMeter tm;
        tm.reset();
        tm.start();
        CWindow* wins = NULL;
        int lwin = -1;
        wins = detect_faces(detector, img.data, img.rows, img.cols, &lwin);
        tm.stop();
        std::cout << "Image: " << i << std::endl;
        std::cout << "Time Cost: " << tm.getTimeMilli() << " ms" << std::endl;

        for (int i = 0; i < lwin; i++)
        {
            for (int p = 0; p < kFeaturePoints; p++)
            {
                cv::Point pt(wins[i].points[p].x,wins[i].points[p].y);
                cv::circle(img, pt, 2, RED, -1);
            }
        }

        free(wins);
        cv::imshow("PCN", img);
        cv::waitKey();
    }

    cv::destroyAllWindows();
    free_detector(detector);
    return 0;
}
