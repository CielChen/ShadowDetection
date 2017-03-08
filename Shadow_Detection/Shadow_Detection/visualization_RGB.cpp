/*
------------------------------------------------
Author: CIEL
Date: 2017/01/16
Function: RGB�ռ�ͼƬ���ӻ�
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
	//IplImage:ͼ�����
	IplImage *img=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��
	
	//��˹�˲�����ƽ��ͼ��
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *channel_r=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);  //cvCreateImage����ͼ�����cvGetSizeͼ��ߴ磬IPL_DEPTH_8U�޷���8λ������0~255����1�Ҷ�ͼ
	IplImage *channel_g=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage *channel_b=cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage *img_r = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);  //3��ɫͼ��ͨ����Ϊ3��RGB��
	IplImage *img_g = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage *img_b = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	cvSplit(img, channel_b, channel_g, channel_r, NULL);  //cvSplit����ͼ��ͨ�������������˳���������
	cvMerge(channel_b, 0, 0, 0, img_b);  //cvMerge�ϲ�ͨ����ʵ�ֲ�ɫͼ����ʾ��Ҳ�ǰ���BGR��˳���������
	cvMerge(0, channel_g, 0, 0, img_g);
	cvMerge(0, 0, channel_r, 0, img_r);

	cvNamedWindow("Image_RGB", CV_WINDOW_AUTOSIZE);  //cvNamedWindowͼ�񴰿�
	cvNamedWindow("Image_R", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_G", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_B", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_RGB", img);  //cvShowImage��ʾͼ��
	cvShowImage("Image_R", img_r);
	cvShowImage("Image_G", img_g);
	cvShowImage("Image_B", img_b);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_Rgb.jpg",img);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_R.jpg",img_r);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_G.jpg",img_g);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\RGB\\Img_B.jpg",img_b);

	cvWaitKey(0);  //cvWaitKey������ͣ���ȴ��û�����һ����������
	cvReleaseImage(&img);
	cvReleaseImage(&img_r);
	cvReleaseImage(&img_g);
	cvReleaseImage(&img_b);
	cvDestroyAllWindows();

	return 0;
}