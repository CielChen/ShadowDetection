/*
------------------------------------------------
Author: CIEL
Date: 2017/01/17
Function: LAB�ռ�ͼƬ���ӻ�
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

//LAB�ռ�ͼƬ���ӻ�
int visualization_LAB()
{
	IplImage *img=cvLoadImage("F:\\Code\\Shadow Detection\\test.jpg",1);  //cvLoadImage��ȡͼ��
	
	//��˹�˲�����ƽ��ͼ��
	cvSmooth(img, img, CV_GAUSSIAN, 3, 0, 0, 0);

	IplImage *LABimg=cvCreateImage(cvGetSize(img), 8, img->nChannels); //����һ��LAB��ɫģʽ�Ŀռ����洢ת�����ͼ�����Ϊ8λ��ͨ������ԭͼ��ͬ

	cvCvtColor(img, LABimg, CV_BGR2Lab);  //��ͼ��img��RGB�ռ�ת��LAB�ռ�

	int step, step_l, channels, cd, cdlab, l, a, b;
	uchar *data_lab, *data_l, *data_a, *data_b;

	IplImage *lab_l= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //����Lͼ��
	IplImage *lab_a= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //����aͼ��
	IplImage *lab_b= cvCreateImage(cvGetSize(LABimg), LABimg->depth, 1);  //����bͼ��

	step=LABimg->widthStep;  //step�洢ͬ��������֮��ı�����
	channels=LABimg->nChannels;  //ͨ����
	data_lab=(uchar*)LABimg->imageData;  //data_lab�洢ָ��LABͼ�����ݵ�ָ��
	step_l=lab_l->widthStep;   //step_lΪ��ͼ���������֮��ı���
	data_l=(uchar*)lab_l->imageData;  //�洢ָ����ͼ�������ָ��
	data_a=(uchar*)lab_a->imageData; 
	data_b=(uchar*)lab_b->imageData; 

	for(int i=0; i<LABimg->height; i++){
		for(int j=0; j<LABimg->width; j++){
			cd= i*step + j*channels;  //����LABͼ�����ݵ�λ��
			cdlab= i*step_l + j;  //����L/a/b��ͼ�����ݴ洢��λ��

			l=data_lab[cd];
			a=data_lab[cd+1];
			b=data_lab[cd+2];

			//L����
			data_l[cdlab]=l;

			//a����
			data_a[cdlab]=a;

			//b����
			data_b[cdlab]=b;
		}
	}

	cvNamedWindow("Image_LAB", CV_WINDOW_AUTOSIZE);  //cvNamedWindowͼ�񴰿�
	cvNamedWindow("Image_L", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_A", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Image_B", CV_WINDOW_AUTOSIZE);
	cvShowImage("Image_LAB", LABimg);  //cvShowImage��ʾͼ��
	cvShowImage("Image_L", lab_l);
	cvShowImage("Image_A", lab_a);
	cvShowImage("Image_B", lab_b);

	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_LAB.jpg",LABimg);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_L.jpg",lab_l);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_A.jpg",lab_a);
	cvSaveImage("F:\\Code\\Shadow Detection\\Data\\Color Space\\LAB\\Img_B.jpg",lab_b);

	cvWaitKey(0);  //cvWaitKey������ͣ���ȴ��û�����һ����������
	cvReleaseImage(&LABimg);
	cvReleaseImage(&lab_l);
	cvReleaseImage(&lab_a);
	cvReleaseImage(&lab_b);
	cvDestroyAllWindows();

	return 0;
}
