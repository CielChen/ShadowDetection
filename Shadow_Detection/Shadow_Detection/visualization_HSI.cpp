/*
------------------------------------------------
Author: CIEL
Date: 2017/01/16
Function: HSI空间图片可视化
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
#include "visualization_HSI.h"

using namespace cv;
using namespace std;

int min(int a, int b, int c)
{
	int m=a;
	if(m>b)
		m=b;
	if(m>c)
		m=c;

	return m;
}

//HSI各分量可视化
int visualization_H_S_I()
{
	int step, step_hsi, channels, cd, cdhsi, b, g, r;
	uchar *data, *data_i, *data_s, *data_h;
	int i,j;
	double min_rgb, add_rgb, theta, den, num;

	IplImage *frame=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像

	//高斯滤波，以平滑图像
	cvSmooth(frame, frame, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *hsi_i= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建I（亮度）图像
	IplImage *hsi_s= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建S（饱和度）图像
	IplImage *hsi_h= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //创建H（色彩）图像
	
	step=frame->widthStep;  //step存储同列相邻行之间的比特数
	channels=frame->nChannels;  //通道数
	data=(uchar*)frame->imageData;  //data存储指向图像数据的指针
	step_hsi=hsi_i->widthStep;   //step_hsi亮度图像的相邻行之间的比特
	data_i=(uchar*)hsi_i->imageData;  //存储指向子图像的数据指针
	data_s=(uchar*)hsi_s->imageData; 
	data_h=(uchar*)hsi_h->imageData; 
	
	for(i=0; i<frame->height; i++){
		for(j=0; j<frame->width; j++){
			cd= i*step + j*channels;  //计算原图像数据的位置
			cdhsi= i*step_hsi + j;  //计算H/S/I子图像数据存储的位置

			b=data[cd];
			g=data[cd+1];
			r=data[cd+2];

			//I分量
			data_i[cdhsi]=(int)( (r+g+b)/3 );

			//S分量，范围[0,255]
			min_rgb=min(r,g,b);  //取最小值
			add_rgb=r+g+b;
			data_s[cdhsi]=(int)(255-765*min_rgb/add_rgb);

			//H分量
			num=0.5*( (r-g)+(r-b) );
			den=sqrt( (double)( (r-g)*(r-g) + (r-b)*(g-b) ) );
			if(den==0)
				den=0.01;
			theta=acos(num/den);

			if(b<=g)
				data_h[cdhsi]=(int)(theta*255/(2*3.14));
			else
				data_h[cdhsi]=(int)(255-theta*255/(2*3.14));
			if(data_s[cdhsi]==0)
				data_h[cdhsi]=0;
		}
	}

	cvNamedWindow("Img_H", 1);
	cvNamedWindow("Img_S", 1);
	cvNamedWindow("Img_I", 1);

	cvShowImage("Img_H", hsi_h);
	cvShowImage("Img_S", hsi_s);
	cvShowImage("Img_I", hsi_i);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSI\\Img_H.jpg", hsi_h);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSI\\Img_S.jpg", hsi_s);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSI\\Img_I.jpg", hsi_i);
	
	waitKey(0);
	cvDestroyAllWindows();
	return 0;
}

//将HSI空间的三个分量组合起来，便于显示
IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage* HSI_Image=cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);

	for(int i=0; i<HSI_Image->height; i++){
		for(int j=0; j<HSI_Image->width; j++){
			double d=cvmGet(HSI_H, i, j);  //cvmGet对于浮点型的单通道矩阵，取出元素[i][j]
			int b=(int)(d*255/360);
			d=cvmGet(HSI_S, i, j);
			int g=(int)(d*255);
			d=cvmGet(HSI_I, i, j);
			int r=(int)(d*255);

			cvSet2D(HSI_Image, i, j, cvScalar(b,g,r) );  //cvSet2D设置图像位置像素坐标的像素值，cvScalar用来存放像素值，最多4个通道
		}
	}

	return HSI_Image;
}

//HSI空间可视化
int visualization_HSI()
{
	IplImage *img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage读取图像
	
	//高斯滤波，以平滑图像
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	//3个HSI空间数据矩阵
	CvMat* HSI_H=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* HSI_S=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* HSI_I=cvCreateMat(img->height, img->width, CV_32FC1);

	//原始图像数据指针，HSI矩阵数据指针
	uchar* data;

	//RGB分量
	typedef unsigned char byte;
	byte img_r, img_g, img_b;
	byte min_rgb;  //RGB分量中的最小值
	//HSI分量
	float fHue, fSaturation, fIntensity;

	for(int i=0; i<img->height; i++){
		for(int j=0; j<img->width; j++){
			data=cvPtr2D(img, i, j, 0);  //cvPtr2D访问矩阵中的[i][j]元素
			img_b=*data;
			data++;
			img_g=*data;
			data++;
			img_r=*data;

			//I分量[0,1]
			fIntensity=(float)((img_b+ img_g+ img_r)/3)/255;

			//得到RGB分量中的最小值
			float fTemp= img_r < img_g ? img_r : img_g;
			min_rgb= fTemp < img_b ? fTemp : img_b;
			//S分量[0,1]
			fSaturation= 1- (float)(3*min_rgb)/(img_r+ img_g+ img_b);

			//计算theta角
			float numerator=(img_r- img_g+ img_r- img_b)/2;
			float denominator=sqrt( pow((img_r-img_g),2) + (img_r-img_b)*(img_g-img_b) );

			//H分量
			if(denominator!=0){
				float theta=acos(numerator/denominator)*180/3.14;  //acos反余弦

				if(img_b<=img_g)
					fHue=theta;
				else
					fHue=360-theta;
			}
			else
				fHue=0;

			//赋值
			cvmSet(HSI_H, i, j, fHue);
			cvmSet(HSI_S, i, j, fSaturation);
			cvmSet(HSI_I, i, j, fIntensity);
		}
	}

	IplImage* HSI_Image=catHSImage(HSI_H, HSI_S, HSI_I);

	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);  //cvNamedWindow图像窗口
	cvNamedWindow("Image_HSI", CV_WINDOW_AUTOSIZE);  
	cvShowImage("img", img);
	cvShowImage("HSI Color Model", HSI_Image);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\HSI\\Img_HSI.jpg",HSI_Image);
	
	cvWaitKey(0);

	cvReleaseImage(&img);
	cvReleaseImage(&HSI_Image);
	cvReleaseMat(&HSI_H);
	cvReleaseMat(&HSI_S);
	cvReleaseMat(&HSI_I);

	cvDestroyAllWindows();

	visualization_H_S_I();  //HSI各分量可视化

	return 0;
}

