/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: LAB空间图片可视化
------------------------------------------------
*/

#include <core/affine.hpp>
#include <highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <stdio.h>
#include <math.h>
#include <cxcore.h>
#include <cvaux.h>
#include <opencv2/opencv.hpp>
#include "visualization_LAB.h"

using namespace cv;
using namespace std;

//LAB空间图片可视化
int visualization_LAB()
{
	IplImage *img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *LABimg=cvCreateImage(cvGetSize(img), 8, img->nChannels); //开辟一个LAB颜色模式的空间来存储转化后的图像，深度为8位，通道数与原图相同

	cvCvtColor(img, LABimg, CV_BGR2Lab);  //将图像img从RGB空间转到LAB空间

	int step, step_l, channels, cd, cdlab, l, a, b;
	uchar *data_lab, *data_l, *data_a, *data_b;

	IplImage *lab_l= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //创建L图像
	IplImage *lab_a= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //创建a图像
	IplImage *lab_b= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //创建b图像

	step=LABimg->widthStep;  //step存储同列相邻行之间的比特数
	channels=LABimg->nChannels;  //通道数
	data_lab=(uchar*)LABimg->imageData;  //data_lab存储指向LAB图像数据的指针
	step_l=lab_l->widthStep;   //step_l为子图像的相邻行之间的比特
	data_l=(uchar*)lab_l->imageData;  //存储指向子图像的数据指针
	data_a=(uchar*)lab_a->imageData; 
	data_b=(uchar*)lab_b->imageData; 

	for(int i=0; i<LABimg->height; i++){
		for(int j=0; j<LABimg->width; j++){
			cd= i*step + j*channels;  //计算LAB图像数据的位置
			cdlab= i*step_l + j;  //计算L/a/b子图像数据存储的位置

			l=data_lab[cd];
			a=data_lab[cd+1];
			b=data_lab[cd+2];

			//L分量
			data_l[cdlab]=l;

			//a分量
			data_a[cdlab]=a;

			//b分量
			data_b[cdlab]=b;
		}
	}

	cvNamedWindow("Image_LAB", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_L", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_A", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_B", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_LAB", LABimg);  //cvShowImage显示图像
	cvShowImage("Image_L", lab_l);
	cvShowImage("Image_A", lab_a);
	cvShowImage("Image_B", lab_b);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_LAB.jpg",LABimg);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_L.jpg",lab_l);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_A.jpg",lab_a);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_B.jpg",lab_b);

	cvWaitKey(0);  //cvWaitKey程序暂停，等待用户触发一个按键操作
	cvReleaseImage(&LABimg);
	cvReleaseImage(&lab_l);
	cvReleaseImage(&lab_a);
	cvReleaseImage(&lab_b);
	cvDestroyAllWindows();

	return 0;
}
