/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: YCbCr空间图片可视化
------------------------------------------------
*/

#include <core/affine.hpp>
#include <highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <stdio.h>
#include <ctype.h>
#include <cxcore.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include "visualization_YCbCr.h"

using namespace cv;
using namespace std;

//YCbCr空间图片可视化
int visualization_YCbCr()
{
	IplImage *img, *YCbCr, *Y, *Cb, *Cr;
	img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	YCbCr=cvCreateImage(cvGetSize(img), 8, 3);  //为YCbCr图像申请空间
	Y=cvCreateImage(cvGetSize(img), 8, 1);  //Y通道
	Cb=cvCreateImage(cvGetSize(img), 8, 1);  //Cb通道
	Cr=cvCreateImage(cvGetSize(img), 8, 1);  //Cr通道

	cvCvtColor(img, YCbCr, CV_BGR2YCrCb);  //将RGB转换为YCbCr

	cvSplit(YCbCr, Y, 0, 0, 0);  //分离三个通道
	cvSplit(YCbCr, 0, Cb, 0, 0);  
	cvSplit(YCbCr, 0, 0, Cr, 0);  

	cvNamedWindow("Image_YCbCr", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_Y", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_Cb", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_Cr", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_YCbCr", YCbCr);  //cvShowImage显示图像
	cvShowImage("Image_Y", Y);
	cvShowImage("Image_Cb", Cb);
	cvShowImage("Image_Cr", Cr);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_YCbCr.jpg",YCbCr);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Y.jpg",Y);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Cb.jpg",Cb);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Cr.jpg",Cr);

	cvWaitKey(0);  //cvWaitKey程序暂停，等待用户触发一个按键操作
	cvReleaseImage(&YCbCr);
	cvReleaseImage(&Y);
	cvReleaseImage(&Cb);
	cvReleaseImage(&Cr);
	cvDestroyAllWindows();
	return 0;
}