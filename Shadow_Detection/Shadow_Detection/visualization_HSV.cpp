/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: HSV�ռ�ͼƬ���ӻ�
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

//HSV�ռ�ͼƬ���ӻ�
int visualization_HSV()
{
	IplImage *img, *hsv, *hue, *saturation, *value;
	img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��
	
	//��˹�˲�����ƽ��ͼ��
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	hsv=cvCreateImage(cvGetSize(img), 8, 3);  //ΪHSVͼ������ռ�
	hue=cvCreateImage(cvGetSize(img), 8, 1);  //H��ɫ����ͨ��
	saturation=cvCreateImage(cvGetSize(img), 8, 1);  //S�����Ͷȣ�ͨ��
	value=cvCreateImage(cvGetSize(img), 8, 1);  //V�����ȣ�ͨ��

	cvCvtColor(img, hsv, CV_BGR2HSV);  //��RGBת��ΪHSV

	cvSplit(hsv, hue, 0, 0, 0);  //��������ͨ��
	cvSplit(hsv, 0, saturation, 0, 0);  
	cvSplit(hsv, 0, 0, value, 0);  

	cvNamedWindow("Image_HSV", CV_WINDOW_AUTOSIZE);  //cvNamedWindowͼ�񴰿�
	cvNamedWindow("Image_H", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_S", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_V", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_HSV", hsv);  //cvShowImage��ʾͼ��
	cvShowImage("Image_H", hue);
	cvShowImage("Image_S", saturation);
	cvShowImage("Image_V", value);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Image_HSV.jpg",hsv);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_H.jpg",hue);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_S.jpg",saturation);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSV\\Img_V.jpg",value);

	cvWaitKey(0);  //cvWaitKey������ͣ���ȴ��û�����һ����������
	cvReleaseImage(&hsv);
	cvReleaseImage(&hue);
	cvReleaseImage(&saturation);
	cvReleaseImage(&value);
	cvDestroyAllWindows();
	return 0;
}