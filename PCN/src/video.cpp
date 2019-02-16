#include "PCN.h"

int main(int argc, char **argv)
{
    //PCN* detector = (PCN*) init_detector(30,1.414,0.37,0.43,0.97,30,0.95,1,0.9,0.6);
    PCN* detector = (PCN*) init_detector(15,1.414,0.37,0.43,0.97,15,0.6,1,0.9,0.3);
    cv::VideoCapture capture;
    if (argc >1)
	    capture.open(argv[1]);
    else
	    capture.open(0);

    if (!capture.isOpened())
	    return 0;

    cv::Mat img;
    cv::TickMeter tm;
    while (1)
    {
        bool ret = capture.read(img);
	if (!ret)
		break;
        tm.reset();
        tm.start();
	
	Window *wins= NULL;
	size_t lwins = 0;
	wins = detect_faces(detector,img.data,img.rows, img.cols, &lwins);

        tm.stop();
        int fps = 1000.0 / tm.getTimeMilli();
        std::stringstream ss;
        ss << std::setw(4) << fps;
        cv::putText(img, std::string("PCN:") + ss.str() + "FPS",
                    cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));
        for (int i = 0; i < lwins; i++)
        {
            DrawFace(img, wins[i]);
        }
	free(wins);
        cv::imshow("PCN", img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    capture.release();
    cv::destroyAllWindows();
    free_detector(detector);
    return 0;
}
