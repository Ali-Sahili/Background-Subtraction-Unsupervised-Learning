#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "DistanceUtils.h"


class LBSP : public cv::DescriptorExtractor {
public:

	LBSP(size_t nThreshold);
	LBSP(float fRelThreshold, size_t nThresholdOffset=0);

	virtual ~LBSP();

	virtual void setReference(const cv::Mat&);

	virtual int descriptorSize() const;

	virtual int descriptorType() const;

	virtual bool isUsingRelThreshold() const;

	virtual float getRelThreshold() const;

	virtual size_t getAbsThreshold() const;

	//similar to DescriptorExtractor::compute(const cv::Mat& image, ...)
	void compute2(	const cv::Mat& oImage, 
                      	std::vector<cv::KeyPoint>& voKeypoints, 
                        cv::Mat& oDescriptors) const;

	void compute2(	const std::vector<cv::Mat>& voImageCollection, 
                      	std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, 
                        std::vector<cv::Mat>& voDescCollection) const;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
	//(1-channel version)
	inline static void computeGrayscaleDescriptor(	const cv::Mat& oInputImg, 
                                                      	const uchar _ref, 
                                                      	const int _x, 
                                                      	const int _y, 
                                                      	const size_t _t, 
                                                      	ushort& _res) 
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);

		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;

		#include "LBSP_16bits_dbcross_1ch.i"
	}

	//(3-channels version)
	inline static void computeRGBDescriptor(  const cv::Mat& oInputImg, 
                                                  const uchar* const _ref,  
                                                  const int _x, 
                                                  const int _y, 
                                                  const size_t* const _t, 
                                                  ushort* _res) 
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);

		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;

		#include "LBSP_16bits_dbcross_3ch3t.i"
	}

	//(3-channels version)
	inline static void computeRGBDescriptor(  const cv::Mat& oInputImg, 
                                                  const uchar* const _ref,  
                                                  const int _x, 
                                                  const int _y, 
                                                  const size_t _t, 
                                                  ushort* _res) 
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);

		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;

		#include "LBSP_16bits_dbcross_3ch1t.i"
	}

	//(1-channel-RGB version)
	inline static void computeSingleRGBDescriptor(  const cv::Mat& oInputImg, 
                                                        const uchar _ref, 
                                                        const int _x, 
                                                        const int _y, 
                                                        const size_t _c, 
                                                        const size_t _t, 
                                                        ushort& _res) 
	{
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3 && _c<3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);

		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;

		#include "LBSP_16bits_dbcross_s3ch.i"
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

	//used to reshape a descriptors matrix to its input image size via their keypoint locations
	static void reshapeDesc(cv::Size oSize, const std::vector<cv::KeyPoint>& voKeypoints, const cv::Mat& oDescriptors, cv::Mat& oOutput);

	//used to illustrate the difference between two descriptor images
	static void calcDescImgDiff(const cv::Mat& oDesc1, const cv::Mat& oDesc2, cv::Mat& oOutput, bool bForceMergeChannels=false);

	//used to filter out bad keypoints --> they're too close to the image border
	static void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize);
	
        //used to filter out bad pixels in a ROI --> they're too close to the image border
	static void validateROI(cv::Mat& oROI);
	
        //specifies the pixel size of the pattern used (width and height)
	static const size_t PATCH_SIZE = 5;

	//specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
	static const size_t DESC_SIZE = 2;

protected:

	const bool m_bOnlyUsingAbsThreshold;
	const float m_fRelThreshold;
	const size_t m_nThreshold;
	cv::Mat m_oRefImage;
};



