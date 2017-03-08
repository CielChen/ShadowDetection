/*
------------------------------------------------
Author: CIEL
Date: 2017/01/16
Function: RGB空间图片可视化
------------------------------------------------
*/

#include <core/affine.hpp>
#include <highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <cvaux.h>
#include "visualization_RGB.h"

using namespace cv;
using namespace std;


int visualization_RGB()
{
	//IplImage:图像对象
	IplImage *img=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *channel_r=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);  //cvCreateImage创建图像对象：cvGetSize图像尺寸，IPL_DEPTH_8U无符号8位整数（0~255），1灰度图
	IplImage *channel_g=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage *channel_b=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage *img_r = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);  //3彩色图，通道数为3（RGB）
	IplImage *img_g = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage *img_b = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	cvSplit(img, channel_b, channel_g, channel_r, NULL);  //cvSplit分离图像通道，分离出来的顺序是逆序的
	cvMerge(channel_b, 0, 0, 0, img_b);  //cvMerge合并通道，实现彩色图像显示，也是按照BGR的顺序来输入的
	cvMerge(0, channel_g, 0, 0, img_g);
	cvMerge(0, 0, channel_r, 0, img_r);

	cvNamedWindow("Image_RGB", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_R", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_G", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_B", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_RGB", img);  //cvShowImage显示图像
	cvShowImage("Image_R", img_r);
	cvShowImage("Image_G", img_g);
	cvShowImage("Image_B", img_b);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_Rgb.jpg",img);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_R.jpg",img_r);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_G.jpg",img_g);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_B.jpg",img_b);

	cvWaitKey(0);  //cvWaitKey程序暂停，等待用户触发一个按键操作
	cvReleaseImage(&img);
	cvReleaseImage(&img_r);
	cvReleaseImage(&img_g);
	cvReleaseImage(&img_b);
	cvDestroyAllWindows();

	return 0;
}