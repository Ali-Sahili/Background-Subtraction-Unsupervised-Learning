#include "FeatureTracker.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{

	FeatureTracker tracker = FeatureTracker(35, 18, 2, 18, 5.0/*5.0*/, 0.0/*0.05*/, 0.05, 1.0, 2, 200, 10.0, 1.0,0.0);

	VideoCapture cap("/home/ali/Desktop/Research/Works_MainTopics/varna_20190125_153327_0_900/varna_20190125_153327_0_900.mp4");
	
	if(!cap.isOpened()){
    		cout << "Error opening video stream or file" << endl;
    		return -1;
  	}

	while(1)
	{
 
		Mat frame;
		Mat output;

		cap >> frame;
  
		if (frame.empty())
			break;
	 
		tracker.process(frame, output);

		Mat dest(frame.rows,frame.cols, CV_8UC3);
		dest = frame.clone();
		std::vector<cv::Mat> channels;
    		cv::split(dest, channels);

		std::cout <<dest.size()<<endl;
		for(int i=0; i<frame.rows; i++)
			for(int j=0; j<frame.cols; j++)
				if(output.at<uchar>(i,j) == 0){
					channels[0].at<uchar>(i,j) = 0;
					channels[1].at<uchar>(i,j) = 0;
					channels[2].at<uchar>(i,j) = 0;
					//dest.at<uchar>(i,j,1) = 0;
					//dest.at<uchar>(i,j,2) = 0;
				}

		cv::merge(channels,dest);

		imshow( "Frame", frame );
		imshow( "Mask", output );
		imshow( "Output", dest );

		char c=(char)waitKey(25);
		if(c==27)
			break;
	}

	// read image frame
	

  	cap.release();
  	destroyAllWindows();

	return 0;
}






/*
#include "FeatureTracker.h"

int main()
{

	FeatureTracker tracker = FeatureTracker(35, 18, 2, 18, 5.0, 0.05, 0.05, 1.0, 2, 200, 10.0, 1.0);
	
	// read image frame
	//cv::Mat output;
	//tracker->process(frame, output);

	// show results

	return 0;
}

*/
