/*
------------------------------------------------
Author: CIEL
Date: 2017/01/18
Function: C1C2C3�ռ�ͼƬ���ӻ�
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

//C1C2C3���������ӻ�
int visualization_C1_C2_C3()
{
	int step, step_c1c2c3, channels, cd, cdc1c2c3, b, g, r;
	uchar *data, *data_c1, *data_c2, *data_c3;
	int i,j;

	IplImage *frame=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��

	//��˹�˲�����ƽ��ͼ��
	cvSmooth(frame, frame, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *c1c2c3_c1= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //����C1ͼ��
	IplImage *c1c2c3_c2= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //����C2ͼ��
	IplImage *c1c2c3_c3= cvCreateImage(cvGetSize(frame), frame->depth, 1);  //����C3ͼ��
	
	step=frame->widthStep;  //step�洢ͬ��������֮��ı�����
	channels=frame->nChannels;  //ͨ����
	data=(uchar*)frame->imageData;  //data�洢ָ��ͼ�����ݵ�ָ��
	step_c1c2c3=c1c2c3_c1->widthStep;   //step_c1c2c3:C1/C2/C3ͼ���������֮��ı���
	data_c1=(uchar*)c1c2c3_c1->imageData;  //�洢ָ����ͼ�������ָ��
	data_c2=(uchar*)c1c2c3_c1->imageData;  
	data_c3=(uchar*)c1c2c3_c1->imageData; 
	
	for(i=0; i<frame->height; i++){
		for(j=0; j<frame->width; j++){
			cd= i*step + j*channels;  //����ԭͼ�����ݵ�λ��
			cdc1c2c3= i*step_c1c2c3 + j;  //����C1/C2/C3��ͼ�����ݴ洢��λ��

			b=data[cd];
			g=data[cd+1];
			r=data[cd+2];

			//C1���� [0,255]
			data_c1[cdc1c2c3]=atan(g / max_ab(r,b) ) * 255;

			//C2���� [0,255]
			data_c2[cdc1c2c3]=atan(r / max_ab(g,b) ) * 255;

			//C3���� [0,255]
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

//��C1C2C3�ռ�������������������������ʾ
IplImage* catC1C2C3Image(CvMat* C1C2C3_C1, CvMat* C1C2C3_C2, CvMat* C1C2C3_C3)
{
	IplImage* C1C2C3_Image=cvCreateImage(cvGetSize(C1C2C3_C1), IPL_DEPTH_8U, 3);

	for(int i=0; i<C1C2C3_Image->height; i++){
		for(int j=0; j<C1C2C3_Image->width; j++){
			double d=cvmGet(C1C2C3_C1, i, j);  //cvmGet���ڸ����͵ĵ�ͨ������ȡ��Ԫ��[i][j]
			int b=(int)(d*255);   //[0,255]
			d=cvmGet(C1C2C3_C2, i, j);
			int g=(int)(d*255);  //[0,255]
			d=cvmGet(C1C2C3_C3, i, j);
			int r=(int)(d*255);   //[0,255]

			cvSet2D(C1C2C3_Image, i, j, cvScalar(b,g,r) );  //cvSet2D����ͼ��λ���������������ֵ��cvScalar�����������ֵ�����4��ͨ��
		}
	}

	return C1C2C3_Image;
}

//C1C2C3�ռ���ӻ�
int visualization_C1C2C3()
{
	IplImage *img=cvLoadImage("G:\\Code-Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��
	
	//��˹�˲�����ƽ��ͼ��
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	//3��C1C2C3�ռ����ݾ���
	CvMat* C1C2C3_C1=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* C1C2C3_C2=cvCreateMat(img->height, img->width, CV_32FC1);
	CvMat* C1C2C3_C3=cvCreateMat(img->height, img->width, CV_32FC1);

	//ԭʼͼ������ָ�룬C1C2C3��������ָ��
	uchar* data;

	//RGB����
	typedef unsigned char byte;
	byte img_r, img_g, img_b;
	//C1C2C3����
	float fC1, fC2, fC3;

	for(int i=0; i<img->height; i++){
		for(int j=0; j<img->width; j++){
			data=cvPtr2D(img, i, j, 0);  //cvPtr2D���ʾ����е�[i][j]Ԫ��
			img_b=*data;
			data++;
			img_g=*data;
			data++;
			img_r=*data;

			//C1����
			fC1=atan(img_g / max_ab(img_r,img_b) );

			//C2����
			fC2=atan(img_r / max_ab(img_g,img_b) );
			
			//C3����
			fC3=atan(img_b / max_ab(img_r,img_g) );

			//��ֵ
			cvmSet(C1C2C3_C1, i, j, fC1);
			cvmSet(C1C2C3_C2, i, j, fC2);
			cvmSet(C1C2C3_C3, i, j, fC3);
		}
	}

	IplImage* C1C2C3_Image=catC1C2C3Image(C1C2C3_C1, C1C2C3_C2, C1C2C3_C3);

	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);  //cvNamedWindowͼ�񴰿�
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

	visualization_C1_C2_C3();  //C1C2C3���������ӻ�

	return 0;
}
