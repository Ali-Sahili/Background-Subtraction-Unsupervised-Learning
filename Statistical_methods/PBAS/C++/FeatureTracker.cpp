
#include "FeatureTracker.h"
#include <opencv2/imgproc.hpp>


void insertionSort(int window[])
{
    int temp, i , j;
    for(i = 0; i < 9; i++){
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--){
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}
 
void MedianFilter(cv::Mat &src, cv::Mat &dst)
{

 
      //create a sliding window of size 9
      int window[9];
 
        dst = src.clone();
        for(int y = 0; y < src.rows; y++)
            for(int x = 0; x < src.cols; x++)
                dst.at<uchar>(y,x) = 0.0;
 
        for(int y = 1; y < src.rows - 1; y++){
            for(int x = 1; x < src.cols - 1; x++){
 
                // Pick up window element
 
                window[0] = src.at<uchar>(y - 1 ,x - 1);
                window[1] = src.at<uchar>(y, x - 1);
                window[2] = src.at<uchar>(y + 1, x - 1);
                window[3] = src.at<uchar>(y - 1, x);
                window[4] = src.at<uchar>(y, x);
                window[5] = src.at<uchar>(y + 1, x);
                window[6] = src.at<uchar>(y - 1, x + 1);
                window[7] = src.at<uchar>(y, x + 1);
                window[8] = src.at<uchar>(y + 1, x + 1);
 
                // sort the window to find median
                insertionSort(window);
 
                // assign the median to centered element of the matrix
                dst.at<uchar>(y,x) = window[4];
            }
        }
}

FeatureTracker::FeatureTracker( int newN, double newR, int newRaute, int newTemporal, //const for pbas
                  double rT, double rID, double iTS, double dTS, int dTRS, int iTRS, //const for pbas
           double newA, double newB,double gam) // double newLabelThresh, int newNeighour) //const for graphCuts

{
	
	//create a pbas-regulator for every rgb channel
	m_pbas1.initialization(newN,newR,newRaute,newTemporal,newA,newB, rT, rID, iTS, dTS, dTRS,iTRS,gam);
	m_pbas2.initialization(newN,newR,newRaute,newTemporal,newA,newB, rT, rID, iTS, dTS, dTRS,iTRS,gam);
	m_pbas3.initialization(newN,newR,newRaute,newTemporal,newA,newB, rT, rID, iTS, dTS, dTRS,iTRS,gam);



	//shadow detection


}

FeatureTracker::~FeatureTracker(void)
{ 	
}


void FeatureTracker::process(cv:: Mat &frame, cv:: Mat &output) 
{
		//###################################
		//PRE-PROCESSING
		//check if bluring is necessary or beneficial at this point
		
		cv::Mat blurImage;
		cv::GaussianBlur(frame, blurImage, cv::Size(3,3), 3);

		//maybe use a bilateralFilter
		//cv::bilateralFilter(frame, blurImage, 5, 15, 15);
		//###################################


		//color image
		std::vector<cv::Mat> rgbChannels(3);
		cv::split(blurImage, rgbChannels);
		//parallelBackgroundAveraging(&m_pbas1,&m_pbas2,&m_pbas3, &rgbChannels, false, &m_pbasResult);
		
		cv::Mat pbasResult1;

		m_pbas1.process(&rgbChannels.at(0), &pbasResult1);
		//pbasResult1.copyTo(output);



		//rgbChannels.at(0).release();
		//rgbChannels.at(1).release();
		//rgbChannels.at(2).release();
		//rgbChannels.clear();

		//###############################################
		//POST-PROCESSING HERE
		//for the final results in the changedetection-challenge a 9x9 median filter has been applied
		cv::Mat result_post;
		
		medianBlur( pbasResult1, result_post, 7 );

		//MedianFilter(pbasResult1, result_post);

		/*
		int morph_elem = 2;
		int morph_size = 0;
		int morph_operator = 2;

		cv::Mat element = cv::getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
		morphologyEx( pbasResult1, result_post, morph_operator, element );
		*/

		result_post.copyTo(output);

		//###############################################
		//m_pbasResult.copyTo(output);
		//blurImage.release();


}
/*
void FeatureTracker::parallelBackgroundAveraging(PBAS* m_pbas1, PBAS* m_pbas2, PBAS* m_pbas3, std::vector<cv::Mat>* rgb,  bool wGC,cv::Mat * pbasR)
{	
	cv::Mat pbasResult1, pbasResult2, pbasResult3;

	//use parallel computing for speed improvements, but higher cpu-workload
	m_tg.run([m_pbas1, rgb, &pbasResult1](){
		m_pbas1->process(&rgb->at(0), &pbasResult1);
	});

	m_tg.run([m_pbas2, rgb, &pbasResult2](){
			m_pbas2->process(&rgb->at(1), &pbasResult2);
	});

	m_tg.run_and_wait([m_pbas3, rgb, &pbasResult3](){
			m_pbas3->process(&rgb->at(2), &pbasResult3);
	});
	

	//just or all foreground results of each rgb channel
	cv::bitwise_or(pbasResult1, pbasResult3, pbasResult1);
	cv::bitwise_or(pbasResult1, pbasResult2, pbasResult1);
	
	pbasResult1.copyTo(*pbasR);
	
	pbasResult2.release();
	pbasResult3.release();
	pbasResult1.release();
}
*/
void FeatureTracker::process(cv::Mat &)
{
}
