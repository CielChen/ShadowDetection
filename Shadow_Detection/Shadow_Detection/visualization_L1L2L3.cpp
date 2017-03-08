/*
------------------------------------------------
Author: CIEL
Date: 2017/01/18
Function: L1L2L3空间图片可视化
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
#include "visualization_L1L2L3.h"

using namespace cv;
using namespace std;

//L1L2L3各分量可视化
int visualization_L1_L2_L3()
{
	int step, step_L1L2L3, channels, cd, cdL1L2L3, b, g, r;
	uchar *data, *data_L1, *data_L2, *data_L3;
	int i,j;
	float num, den;

	IplImage *frame=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像

	//高斯滤波，以平滑图像
	cvSmooth(frame, frame, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *L1L2L3_L1= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建L1图像
	IplImage *L1L2L3_L2= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建L2图像
	IplImage *L1L2L3_L3= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建L3图像
	
	step=frame->widthStep;  //step存储同列相邻行之间的比特数
	channels=frame->nChannels;  //通道数
	data=(uchar*)frame->imageData;  //data存储指向图像数据的指针
	step_L1L2L3=L1L2L3_L1->widthStep;   //step_L1L2L3:L1/L2/L3图像的相邻行之间的比特
	data_L1=(uchar*)L1L2L3_L1->imageData;  //存储指向子图像的数据指针
	data_L2=(uchar*)L1L2L3_L2->imageData;  
	data_L3=(uchar*)L1L2L3_L3->imageData; 
	
	for(i=0; i<frame->height; i++){
		for(j=0; j<frame->width; j++){
			cd= i*step + j*channels;  //计算原图像数据的位置
			cdL1L2L3= i*step_L1L2L3 + j;  //计算L1/L2/L3子图像数据存储的位置

			b=data[cd];
			g=data[cd+1];
			r=data[cd+2];

			den=pow( (r-g), 2) + pow( (r-b), 2) + pow( (g-b), 2);
			//L1分量 [0,255]
			num=pow( (r-g), 2);
			data_L1[cdL1L2L3]=num/den * 255;

			//L2分量 [0,255]
			num=pow( (r-b), 2);
			data_L2[cdL1L2L3]=num/den * 255;

			//L3分量 [0,255]
			num=pow( (g-b), 2);
			data_L3[cdL1L2L3]=num/den * 255;
		}
	}

	cvNamedWindow("Img_L1", 1);
	cvNamedWindow("Img_L2", 1);
	cvNamedWindow("Img_L3", 1);

	cvShowImage("Img_L1", L1L2L3_L1);
	cvShowImage("Img_L2", L1L2L3_L2);
	cvShowImage("Img_L3", L1L2L3_L3);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\L1L2L3\\Img_L1.jpg", L1L2L3_L1);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\L1L2L3\\Img_L2.jpg", L1L2L3_L2);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\L1L2L3\\Img_L3.jpg", L1L2L3_L3);
	
	waitKey(0);
	cvDestroyAllWindows();
	return 0;
}

//将L1L2L3空间的三个分量组合起来，便于显示
IplImage* catL1L2L3Image(CvMat* L1L2L3_L1, CvMat* L1L2L3_L2, CvMat* L1L2L3_L3)
{
	IplImage* L1L2L3_Image=cvCreateImage(cvGetSize(L1L2L3_L1), IPL_DEPTH_8U, 3);

	for(int i=0; i<L1L2L3_Image->height; i++){
		for(int j=0; j<L1L2L3_Image->width; j++){
			double d=cvmGet(L1L2L3_L1, i, j);  //cvmGet对于浮点型的单通道矩阵，取出元素[i][j]
			int b=(int)(d*255);   //[0,255]
			d=cvmGet(L1L2L3_L2, i, j);
			int g=(int)(d*255);  //[0,255]
			d=cvmGet(L1L2L3_L3, i, j);
			int r=(int)(d*255);   //[0,255]

			cvSet2D(L1L2L3_Image, i, j, cvScalar(b,g,r) );  //cvSet2D设置图像位置像素坐标的像素值，cvScalar用来存放像素值，最多4个通道
		}
	}

	return L1L2L3_Image;
}

//L1L2L3空间可视化
int visualization_L1L2L3()
{
	IplImage *img=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	//3个L1L2L3空间数据矩阵
	CvMat* L1L2L3_L1=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* L1L2L3_L2=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* L1L2L3_L3=cvCreateMat(img->height, img->width, CV_32FC1);

	//原始图像数据指针，L1L2L3矩阵数据指针
	uchar* data;

	//RGB分量
	typedef unsigned char byte;
	byte img_r, img_g, img_b;
	//L1L2L3分量
	float fL1, fL2, fL3;

	float num,den;
	for(int i=0; i<img->height; i++){
		for(int j=0; j<img->width; j++){
			data=cvPtr2D(img, i, j, 0);  //cvPtr2D访问矩阵中的[i][j]元素
			img_b=*data;
			data++;
			img_g=*data;
			data++;
			img_r=*data;

			den=pow( (img_r-img_g), 2) + pow( (img_r-img_b), 2) + pow( (img_g-img_b), 2);
			//L1分量 [0,255]
			num=pow( (img_r-img_g), 2);
			fL1=num/den * 255;

			//L2分量 [0,255]
			num=pow( (img_r-img_b), 2);
			fL2=num/den * 255;
			
			//L3分量 [0,255]
			num=pow( (img_g-img_b), 2);
			fL3=num/den * 255;

			//赋值
			cvmSet(L1L2L3_L1, i, j, fL1);
			cvmSet(L1L2L3_L2, i, j, fL2);
			cvmSet(L1L2L3_L3, i, j, fL3);
		}
	}

	IplImage* L1L2L3_Image=catL1L2L3Image(L1L2L3_L1, L1L2L3_L2, L1L2L3_L3);

	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_L1L2L3", CV_WINDOW_AUTOSIZE);  
	cvShowImage("img", img);
	cvShowImage("L1L2L3 Color Model", L1L2L3_Image);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\L1L2L3\\Image_L1L2L3.jpg",L1L2L3_Image);
	
	cvWaitKey(0);

	cvReleaseImage(&img);
	cvReleaseImage(&L1L2L3_Image);
	cvReleaseMat(&L1L2L3_L1);
	cvReleaseMat(&L1L2L3_L2);
	cvReleaseMat(&L1L2L3_L3);

	cvDestroyAllWindows();

	visualization_L1_L2_L3();  //L1L2L3各分量可视化

	return 0;
}
