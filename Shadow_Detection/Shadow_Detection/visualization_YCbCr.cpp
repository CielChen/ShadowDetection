/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: YCbCr�ռ�ͼƬ���ӻ�
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

//YCbCr�ռ�ͼƬ���ӻ�
int visualization_YCbCr()
{
	IplImage *img, *YCbCr, *Y, *Cb, *Cr;
	img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��
	
	//��˹�˲�����ƽ��ͼ��
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	YCbCr=cvCreateImage(cvGetSize(img), 8, 3);  //ΪYCbCrͼ������ռ�
	Y=cvCreateImage(cvGetSize(img), 8, 1);  //Yͨ��
	Cb=cvCreateImage(cvGetSize(img), 8, 1);  //Cbͨ��
	Cr=cvCreateImage(cvGetSize(img), 8, 1);  //Crͨ��

	cvCvtColor(img, YCbCr, CV_BGR2YCrCb);  //��RGBת��ΪYCbCr

	cvSplit(YCbCr, Y, 0, 0, 0);  //��������ͨ��
	cvSplit(YCbCr, 0, Cb, 0, 0);  
	cvSplit(YCbCr, 0, 0, Cr, 0);  

	cvNamedWindow("Image_YCbCr", CV_WINDOW_AUTOSIZE);  //cvNamedWindowͼ�񴰿�
	cvNamedWindow("Image_Y", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_Cb", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_Cr", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_YCbCr", YCbCr);  //cvShowImage��ʾͼ��
	cvShowImage("Image_Y", Y);
	cvShowImage("Image_Cb", Cb);
	cvShowImage("Image_Cr", Cr);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_YCbCr.jpg",YCbCr);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Y.jpg",Y);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Cb.jpg",Cb);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\YCbCr\\Image_Cr.jpg",Cr);

	cvWaitKey(0);  //cvWaitKey������ͣ���ȴ��û�����һ����������
	cvReleaseImage(&YCbCr);
	cvReleaseImage(&Y);
	cvReleaseImage(&Cb);
	cvReleaseImage(&Cr);
	cvDestroyAllWindows();
	return 0;
}