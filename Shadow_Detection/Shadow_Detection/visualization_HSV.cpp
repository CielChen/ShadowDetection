/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: HSV空间图片可视化
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
#include "visualization_HSV.h"

using namespace cv;
using namespace std;

//HSV空间图片可视化
int visualization_HSV()
{
	IplImage *img, *hsv, *hue, *saturation, *value;
	img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	hsv=cvCreateImage(cvGetSize(img), 8, 3);  //为HSV图像申请空间
	hue=cvCreateImage(cvGetSize(img), 8, 1);  //H（色调）通道
	saturation=cvCreateImage(cvGetSize(img), 8, 1);  //S（饱和度）通道
	value=cvCreateImage(cvGetSize(img), 8, 1);  //V（亮度）通道

	cvCvtColor(img, hsv, CV_BGR2HSV);  //将RGB转换为HSV

	cvSplit(hsv, hue, 0, 0, 0);  //分离三个通道
	cvSplit(hsv, 0, saturation, 0, 0);  
	cvSplit(hsv, 0, 0, value, 0);  

	cvNamedWindow("Image_HSV", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_H", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_S", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_V", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_HSV", hsv);  //cvShowImage显示图像
	cvShowImage("Image_H", hue);
	cvShowImage("Image_S", saturation);
	cvShowImage("Image_V", value);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Image_HSV.jpg",hsv);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_H.jpg",hue);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_S.jpg",saturation);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_V.jpg",value);

	cvWaitKey(0);  //cvWaitKey程序暂停，等待用户触发一个按键操作
	cvReleaseImage(&hsv);
	cvReleaseImage(&hue);
	cvReleaseImage(&saturation);
	cvReleaseImage(&value);
	cvDestroyAllWindows();
	return 0;
}