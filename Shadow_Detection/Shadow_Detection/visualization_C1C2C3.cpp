/*
------------------------------------------------
Author: CIEL
Date: 2017/01/18
Function: C1C2C3空间图片可视化
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
#include "visualization_C1C2C3.h"

using namespace cv;
using namespace std;

int max_ab(int a, int b)
{
	int m=a;
	if(m<b)
		m=b;
	return m;
}

//C1C2C3各分量可视化
int visualization_C1_C2_C3()
{
	int step, step_c1c2c3, channels, cd, cdc1c2c3, b, g, r;
	uchar *data, *data_c1, *data_c2, *data_c3;
	int i,j;

	IplImage *frame=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像

	//高斯滤波，以平滑图像
	cvSmooth(frame, frame, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *c1c2c3_c1= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建C1图像
	IplImage *c1c2c3_c2= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建C2图像
	IplImage *c1c2c3_c3= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建C3图像
	
	step=frame->widthStep;  //step存储同列相邻行之间的比特数
	channels=frame->nChannels;  //通道数
	data=(uchar*)frame->imageData;  //data存储指向图像数据的指针
	step_c1c2c3=c1c2c3_c1->widthStep;   //step_c1c2c3:C1/C2/C3图像的相邻行之间的比特
	data_c1=(uchar*)c1c2c3_c1->imageData;  //存储指向子图像的数据指针
	data_c2=(uchar*)c1c2c3_c1->imageData;  
	data_c3=(uchar*)c1c2c3_c1->imageData; 
	
	for(i=0; i<frame->height; i++){
		for(j=0; j<frame->width; j++){
			cd= i*step + j*channels;  //计算原图像数据的位置
			cdc1c2c3= i*step_c1c2c3 + j;  //计算C1/C2/C3子图像数据存储的位置

			b=data[cd];
			g=data[cd+1];
			r=data[cd+2];

			//C1分量 [0,255]
			data_c1[cdc1c2c3]=atan(g / max_ab(r,b) ) * 255;

			//C2分量 [0,255]
			data_c2[cdc1c2c3]=atan(r / max_ab(g,b) ) * 255;

			//C3分量 [0,255]
			data_c3[cdc1c2c3]=atan(b / max_ab(r,g) ) * 255;
		}
	}

	cvNamedWindow("Img_C1", 1);
	cvNamedWindow("Img_C2", 1);
	cvNamedWindow("Img_C3", 1);

	cvShowImage("Img_C1", c1c2c3_c1);
	cvShowImage("Img_C2", c1c2c3_c2);
	cvShowImage("Img_C3", c1c2c3_c3);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\C1C2C3\\Img_C1.jpg", c1c2c3_c1);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\C1C2C3\\Img_C2.jpg", c1c2c3_c2);
	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\C1C2C3\\Img_C3.jpg", c1c2c3_c3);
	
	waitKey(0);
	cvDestroyAllWindows();
	return 0;
}

//将C1C2C3空间的三个分量组合起来，便于显示
IplImage* catC1C2C3Image(CvMat* C1C2C3_C1, CvMat* C1C2C3_C2, CvMat* C1C2C3_C3)
{
	IplImage* C1C2C3_Image=cvCreateImage(cvGetSize(C1C2C3_C1), IPL_DEPTH_8U, 3);

	for(int i=0; i<C1C2C3_Image->height; i++){
		for(int j=0; j<C1C2C3_Image->width; j++){
			double d=cvmGet(C1C2C3_C1, i, j);  //cvmGet对于浮点型的单通道矩阵，取出元素[i][j]
			int b=(int)(d*255);   //[0,255]
			d=cvmGet(C1C2C3_C2, i, j);
			int g=(int)(d*255);  //[0,255]
			d=cvmGet(C1C2C3_C3, i, j);
			int r=(int)(d*255);   //[0,255]

			cvSet2D(C1C2C3_Image, i, j, cvScalar(b,g,r) );  //cvSet2D设置图像位置像素坐标的像素值，cvScalar用来存放像素值，最多4个通道
		}
	}

	return C1C2C3_Image;
}

//C1C2C3空间可视化
int visualization_C1C2C3()
{
	IplImage *img=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	//3个C1C2C3空间数据矩阵
	CvMat* C1C2C3_C1=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* C1C2C3_C2=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* C1C2C3_C3=cvCreateMat(img->height, img->width, CV_32FC1);

	//原始图像数据指针，C1C2C3矩阵数据指针
	uchar* data;

	//RGB分量
	typedef unsigned char byte;
	byte img_r, img_g, img_b;
	//C1C2C3分量
	float fC1, fC2, fC3;

	for(int i=0; i<img->height; i++){
		for(int j=0; j<img->width; j++){
			data=cvPtr2D(img, i, j, 0);  //cvPtr2D访问矩阵中的[i][j]元素
			img_b=*data;
			data++;
			img_g=*data;
			data++;
			img_r=*data;

			//C1分量
			fC1=atan(img_g / max_ab(img_r,img_b) );

			//C2分量
			fC2=atan(img_r / max_ab(img_g,img_b) );
			
			//C3分量
			fC3=atan(img_b / max_ab(img_r,img_g) );

			//赋值
			cvmSet(C1C2C3_C1, i, j, fC1);
			cvmSet(C1C2C3_C2, i, j, fC2);
			cvmSet(C1C2C3_C3, i, j, fC3);
		}
	}

	IplImage* C1C2C3_Image=catC1C2C3Image(C1C2C3_C1, C1C2C3_C2, C1C2C3_C3);

	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_C1C2C3", CV_WINDOW_AUTOSIZE);  
	cvShowImage("img", img);
	cvShowImage("C1C2C3 Color Model", C1C2C3_Image);

	cvSaveImage("G:\\Code-Shadow Detection\\Data\\Color Space\\C1C2C3\\Img_C1C2C3.jpg",C1C2C3_Image);
	
	cvWaitKey(0);

	cvReleaseImage(&img);
	cvReleaseImage(&C1C2C3_Image);
	cvReleaseMat(&C1C2C3_C1);
	cvReleaseMat(&C1C2C3_C2);
	cvReleaseMat(&C1C2C3_C3);

	cvDestroyAllWindows();

	visualization_C1_C2_C3();  //C1C2C3各分量可视化

	return 0;
}
