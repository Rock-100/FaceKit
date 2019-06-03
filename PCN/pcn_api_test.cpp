#include "PCN_API.h"
int main(int argc, char **argv)
{
	const char *detection_model_path = "./model/PCN.caffemodel";
	const char *pcn1_proto = "./model/PCN-1.prototxt";
	const char *pcn2_proto = "./model/PCN-2.prototxt";
	const char *pcn3_proto = "./model/PCN-3.prototxt";
	const char *tracking_model_path = "./model/PCN-Tracking.caffemodel";
	const char *tracking_proto = "./model/PCN-Tracking.prototxt";

	PCN* detector = (PCN*) init_detector(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto,
			15,1.45,0.5,0.5,0.98,30,0.9,1);
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

		CWindow* wins = NULL;
		int lwin = -1;
		wins = detect_faces(detector,img.data,img.rows, img.cols,&lwin);
		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << std::setw(4) << fps;
		cv::putText(img, std::string("PCN:") + ss.str() + "FPS",
				cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));
		for (int i = 0; i < lwin; i++){
			for (int p=0; p < kFeaturePoints; p++){
				cv::Point pt(wins[i].points[p].x,wins[i].points[p].y);
				cv::circle(img, pt, 2, RED, -1);
			}
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
