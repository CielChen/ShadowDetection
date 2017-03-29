/*
------------------------------------------------
Author: CIEL
Date: 2017/03/01
Function: ��Ӱ���
------------------------------------------------
*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include <limits.h>
#include <cv.h>
#include "shadow_detection.h"

using namespace cv;
using namespace std;

/*
#define WIDTH 1408  //HoloLensͼ����
#define HEIGHT 792  //HoloLensͼ��߶�
*/

/*
#define WIDTH 320  //ͼ����
#define HEIGHT 240  //ͼ��߶�
*/

#define WIDTH 1216  //HoloLens��Ƶ��ͼ���
#define HEIGHT 684  //HoloLens��Ƶ��ͼ�߶�

#define WINDOW_NAME1 "ԭʼͼ����"
#define WINDOW_NAME2 "Ч��ͼ����"

Mat sceneMat, backgroundMat;
Mat chromaticityMat, brightnessMat, localMat, spacialMat, spacialGrayMat;  //�洢ÿ����Ӱ�����

/*//���д����
int sceneRGB_B[WIDTH][HEIGHT],sceneRGB_G[WIDTH][HEIGHT],sceneRGB_R[WIDTH][HEIGHT];  //ǰ��ͼ���RGB����
int backgroundRGB_B[WIDTH][HEIGHT],backgroundRGB_G[WIDTH][HEIGHT],backgroundRGB_R[WIDTH][HEIGHT];  //����ͼ���RGB����
double sceneNorm[WIDTH][HEIGHT],backgroundNorm[WIDTH][HEIGHT];  //ǰ��ͼ��ͱ���ͼ��ÿ������RGB������2����
double cd_B[WIDTH][HEIGHT],cd_G[WIDTH][HEIGHT],cd_R[WIDTH][HEIGHT];  //ɫ�Ȳ��RGB����
double bd_B[WIDTH][HEIGHT],bd_G[WIDTH][HEIGHT],bd_R[WIDTH][HEIGHT];  //���Ȳ��RGB����
double q_B[WIDTH][HEIGHT], q_G[WIDTH][HEIGHT], q_R[WIDTH][HEIGHT]; //RGB����������Qֵ
*/
int sceneRGB_B[HEIGHT][WIDTH],sceneRGB_G[HEIGHT][WIDTH],sceneRGB_R[HEIGHT][WIDTH];  //ǰ��ͼ���RGB����
int backgroundRGB_B[HEIGHT][WIDTH],backgroundRGB_G[HEIGHT][WIDTH],backgroundRGB_R[HEIGHT][WIDTH];  //����ͼ���RGB����
double sceneNorm[HEIGHT][WIDTH],backgroundNorm[HEIGHT][WIDTH];  //ǰ��ͼ��ͱ���ͼ��ÿ������RGB������2����
double cd_B[HEIGHT][WIDTH],cd_G[HEIGHT][WIDTH],cd_R[HEIGHT][WIDTH];  //ɫ�Ȳ��RGB����
double bd_B[HEIGHT][WIDTH],bd_G[HEIGHT][WIDTH],bd_R[HEIGHT][WIDTH];  //���Ȳ��RGB����
double q_B[HEIGHT][WIDTH], q_G[HEIGHT][WIDTH], q_R[HEIGHT][WIDTH]; //RGB����������Qֵ

//ɫ��CD
double cd_m_B, cd_m_G, cd_m_R;  //RGB��������������
double cd_variance_B, cd_variance_G, cd_variance_R;   //RGB���������ķ���
double cd_thresholdH_B, cd_thresholdL_B, cd_thresholdH_G, cd_thresholdL_G, cd_thresholdH_R, cd_thresholdL_R;  //RGB���������ĸߵ���ֵ
//����BD
double bd_m_B, bd_m_G, bd_m_R;  //RGB��������������
double bd_variance_B, bd_variance_G, bd_variance_R;   //RGB���������ķ���
double bd_thresholdH_B, bd_thresholdL_B, bd_thresholdH_G, bd_thresholdL_G, bd_thresholdH_R, bd_thresholdL_R;  //RGB���������ĸߵ���ֵ

struct pixelInformation{   //�ṹ����ÿ�����ص����Ϣ
	int category;  //�ж����ص�����ࡣ0��������1�����壻2����Ӱ
	int initColor_B;  //ԭͼ��RBG��ɫ-B
	int initColor_G;  //ԭͼ��RBG��ɫ-G
	int initColor_R;  //ԭͼ��RBG��ɫ-R
	int revise;  //�жϸ������Ƿ��޸ġ�0��û�б��޸ģ�1�����޸�
};
//struct pixelInformation graph[WIDTH][HEIGHT];
struct pixelInformation graph[HEIGHT][WIDTH];


int chromaticityShadowNum;  //ɫ�Ȳ��⵽����Ӱ������

int g_nThresh=100;
int g_maxThresh=255;
RNG g_rng(12345);
//vector<vector<Point>> g_vContours;
//vector<Vec4i> g_viHierarchy;
void on_ThreshChange(int, void*);

//��������2����
double norm2(int b,int g,int r)
{
	double norm=0;
	norm=sqrt(b*b+g*g+r*r);
	return norm;
}

//����ǰ��ͼ���뱳��ͼ���ɫ�Ȳ�
//���������ǰ��ͼƬ�Ѿ�����Ϊ�����ǻ�ɫ��ǰ���ǻ�ɫ
int chromaticityDiffer()
{
	sceneMat=imread("F:\\Code\\Shadow Detection\\Data\\Foreground\\20170228111043_fore.jpg");  //�ʼ��ȡ����ǰ��ͼ��
	backgroundMat=imread("F:\\Code\\Shadow Detection\\Data\\Background\\20170228111043_back.jpg");  //����ͼ��

	namedWindow("ǰ��ͼ");
	imshow("ǰ��ͼ", sceneMat);
	waitKey(0);

	//����ǰ��ͼ���ÿ�����أ�ע��RGB����������Ҫ����
	for(int i=0;i<sceneMat.rows;i++)
	{
		const Vec3b* scenePoint=sceneMat.ptr <Vec3b>(i);  //Vec3b��һ����Ԫ���������ݽṹ�������ܹ���ʾRGB����������
		for(int j=0;j<sceneMat.cols;j++)
		{
			Vec3b intensity=*(scenePoint+j);
			sceneRGB_B[i][j]=intensity[0];
			sceneRGB_G[i][j]=intensity[1];
			sceneRGB_R[i][j]=intensity[2];

			//��ʼ���ṹ��������𣬳�ʼ����Ϊÿ�����ض��Ǳ���
			graph[i][j].category=0;
			//��ʼ���ṹ����ɫ����ɫͬǰ��ͼ��
			graph[i][j].initColor_B=sceneRGB_B[i][j];
			graph[i][j].initColor_G=sceneRGB_G[i][j];
			graph[i][j].initColor_R=sceneRGB_R[i][j];
		}
	}

	//��������ͼ���ÿ�����أ�ע��RGB����������Ҫ����
	for(int i=0;i<backgroundMat.rows;i++)
	{
		const Vec3b* backgrounPoint=backgroundMat.ptr <Vec3b>(i);  //Vec3b��һ����Ԫ���������ݽṹ�������ܹ���ʾRGB����������
		for(int j=0;j<backgroundMat.cols;j++)
		{
			Vec3b intensity=*(backgrounPoint+j);
			backgroundRGB_B[i][j]=intensity[0];
			backgroundRGB_G[i][j]=intensity[1];
			backgroundRGB_R[i][j]=intensity[2];
		}
	}

	//����ǰ��ͼ��ͱ���ͼ��ÿ������RGB������2����
	for(int i=0;i<backgroundMat.rows;i++)
	{
		for(int j=0;j<backgroundMat.cols;j++)
		{
			sceneNorm[i][j]=norm2(sceneRGB_B[i][j],sceneRGB_G[i][j],sceneRGB_R[i][j]);
			backgroundNorm[i][j]=norm2(backgroundRGB_B[i][j],backgroundRGB_G[i][j],backgroundRGB_R[i][j]);
		}
	}

	//����ǰ��ͼ���뱳��ͼ��ÿ�����ص�ɫ�Ȳע��RGB����������Ҫ����
	//ע�����ܻ���ַ�ĸΪ��������������һ��Ҫ�жϣ������������ĸΪ�㣬������Ϊ����С��������
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(sceneNorm[i][j]==0)
				sceneNorm[i][j]=INT_MIN;
			if(backgroundNorm[i][j]==0)
				backgroundNorm[i][j]=INT_MIN;

			cd_B[i][j]=sceneRGB_B[i][j]/sceneNorm[i][j]-backgroundRGB_B[i][j]/backgroundNorm[i][j];
			cd_G[i][j]=sceneRGB_G[i][j]/sceneNorm[i][j]-backgroundRGB_G[i][j]/backgroundNorm[i][j];
			cd_R[i][j]=sceneRGB_R[i][j]/sceneNorm[i][j]-backgroundRGB_R[i][j]/backgroundNorm[i][j];
		}
	}

	//��ÿ�����ص�CDֵ���浽txt�ļ���
	//B����
	ofstream out_cdB("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_B.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdB<<cd_B[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_cdB<<endl;   //ÿ�������������ӻ���
	}
	//G����
	ofstream out_cdG("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_G.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdG<<cd_G[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_cdG<<endl;   //ÿ�������������ӻ���
	}
	//R����
	ofstream out_cdR("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_R.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdR<<cd_R[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_cdR<<endl;   //ÿ�������������ӻ���
	}
	out_cdB.close();
	out_cdG.close();
	out_cdR.close();


	//����CD������
	int cdNum_B=0, cdNum_G=0, cdNum_R=0; //RGB����������������Ϊ[-0.2,0.2]��CD�ĸ���
	cd_m_B=0;
	cd_m_G=0;
	cd_m_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_B++;
				cd_m_B=cd_m_B+cd_B[i][j];
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_G++;
				cd_m_G=cd_m_G+cd_G[i][j];
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_R++;
				cd_m_R=cd_m_R+cd_R[i][j];
			}
		}
	}
	cd_m_B=cd_m_B/cdNum_B;
	cd_m_G=cd_m_G/cdNum_G;
	cd_m_R=cd_m_R/cdNum_R;

	//����CD�ķ���
	cd_variance_B=0;
	cd_variance_G=0;
	cd_variance_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B������������������Ϊ[-0.2,0.2]��CD
			{
				cd_variance_B=cd_variance_B + pow((cd_B[i][j]-cd_m_B),2);
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G������������������Ϊ[-0.2,0.2]��CD
			{
				cd_variance_G=cd_variance_G + pow((cd_G[i][j]-cd_m_G),2);
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R������������������Ϊ[-0.2,0.2]��CD
			{
				cd_variance_R=cd_variance_R + pow((cd_R[i][j]-cd_m_R),2);
			}
		}
	}
	cd_variance_B=sqrt(cd_variance_B/cdNum_B);
	cd_variance_G=sqrt(cd_variance_G/cdNum_G);
	cd_variance_R=sqrt(cd_variance_R/cdNum_R);

	//����RGB���������ĸߵ���ֵ
	cd_thresholdH_B= cd_m_B + 1.96*cd_variance_B;  //B����
	cd_thresholdL_B= cd_m_B - 1.96*cd_variance_B;
	cd_thresholdH_G= cd_m_G + 1.96*cd_variance_G;  //G����
	cd_thresholdL_G= cd_m_G - 1.96*cd_variance_G;
	cd_thresholdH_R= cd_m_R + 1.96*cd_variance_R;  //R����
	cd_thresholdL_R= cd_m_R - 1.96*cd_variance_R;

	cout<<"--------------------CD������----------------------"<<endl;
	cout<<"[-0.2,0.2]cdNum_B��"<<cdNum_B<<endl;
	cout<<"[-0.2,0.2]cdNum_G��"<<cdNum_G<<endl;
	cout<<"[-0.2,0.2]cdNum_R��"<<cdNum_R<<endl;
	cout<<"CD_B��������"<<cd_m_B<<endl;
	cout<<"CD_G��������"<<cd_m_G<<endl;
	cout<<"CD_R��������"<<cd_m_R<<endl;
	cout<<"CD_B�ķ��"<<cd_variance_B<<endl;
	cout<<"CD_G�ķ��"<<cd_variance_G<<endl;
	cout<<"CD_R�ķ��"<<cd_variance_R<<endl;
	cout<<"B�ķ�����ֵ��"<<cd_thresholdL_B<<"\t"<<cd_thresholdH_B<<endl;
	cout<<"G�ķ�����ֵ��"<<cd_thresholdL_G<<"\t"<<cd_thresholdH_G<<endl;
	cout<<"R�ķ�����ֵ��"<<cd_thresholdL_R<<"\t"<<cd_thresholdH_R<<endl;

	chromaticityMat=sceneMat.clone();   //�����chromaticityMat������sceneMat���γ�һ���µ�ͼ����������໥û��Ӱ��	
	chromaticityShadowNum=0;  //ɫ�Ȳ��⵽����Ӱ������
	//����BD������
	bd_m_B=0;
	bd_m_G=0;
	bd_m_R=0;
	//��ʼ��BD
	for(int i=0;i<chromaticityMat.rows;i++)
	{
		for(int j=0;j<chromaticityMat.cols;j++)
		{
			bd_B[i][j]=0;
			bd_G[i][j]=0;
			bd_R[i][j]=0;
		}
	}
	//ǰ��ͼ�񱳾�Ϊ��ɫ�������ѡ���ǻƣ�����ֻ���������ѡ��������ĳ���������廹����Ӱ
	for(int i=0;i<chromaticityMat.rows;i++)
	{
		for(int j=0;j<chromaticityMat.cols;j++)
		{		
			//��ɫ�Ǳ���:�������������
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)<=30 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)<=30 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-255)<=30 )
				continue;  
			else
			{
				if(cd_B[i][j]>0.2 || cd_G[i][j]>0.2  ||cd_R[i][j]>0.2)  //CD>0.2,��Ϊ����
				{
					chromaticityMat.at<Vec3b>(i,j)[0]=0;   //����Ϊ��ɫ
					chromaticityMat.at<Vec3b>(i,j)[1]=0;
					chromaticityMat.at<Vec3b>(i,j)[2]=255;

					//�޸Ľṹ������Ӧ����Ϣ
					graph[i][j].category=1;
					graph[i][j].initColor_B=0;
					graph[i][j].initColor_G=0;
					graph[i][j].initColor_R=255;
				}
				else
				{
					//CD����ֵ�����ڣ�������Ӱ
					if( (cd_B[i][j]>cd_thresholdL_B && cd_B[i][j]<cd_thresholdH_B) || (cd_G[i][j]>cd_thresholdL_G && cd_G[i][j]<cd_thresholdH_G) || (cd_R[i][j]>cd_thresholdL_R && cd_R[i][j]<cd_thresholdH_R) )
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;  //��ӰΪ��ɫ
						chromaticityMat.at<Vec3b>(i,j)[1]=255;
						chromaticityMat.at<Vec3b>(i,j)[2]=0;

						//�޸Ľṹ������Ӧ����Ϣ
						graph[i][j].category=2;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=255;
						graph[i][j].initColor_R=0;

						//ͳ��ɫ�Ȳ��⵽����Ӱ���ظ���,�������ǵ�BDֵ
						chromaticityShadowNum++;  
						//ע�����ܻ���ַ�ĸΪ��������������һ��Ҫ�жϣ������������ĸΪ�㣬������Ϊ����С��������
						if(backgroundRGB_B[i][j]==0)
							backgroundRGB_B[i][j]=INT_MIN;
						if(backgroundRGB_G[i][j]==0)
							backgroundRGB_G[i][j]=INT_MIN;
						if(backgroundRGB_R[i][j]==0)
							backgroundRGB_R[i][j]=INT_MIN;
						bd_B[i][j]=sceneRGB_B[i][j]/backgroundRGB_B[i][j];
						bd_G[i][j]=sceneRGB_G[i][j]/backgroundRGB_G[i][j];
						bd_R[i][j]=sceneRGB_R[i][j]/backgroundRGB_R[i][j];
						bd_m_B=bd_m_B+bd_B[i][j];  //B����
						bd_m_G=bd_m_G+bd_G[i][j];  //G����
						bd_m_R=bd_m_R+bd_R[i][j];  //R����
					}  
					else
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;   //����Ϊ��ɫ
						chromaticityMat.at<Vec3b>(i,j)[1]=0;
						chromaticityMat.at<Vec3b>(i,j)[2]=255;

						//�޸Ľṹ������Ӧ����Ϣ
						graph[i][j].category=1;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=0;
						graph[i][j].initColor_R=255;
					}

				}
			}

		}
	}
	cout<<"ɫ�Ȳ��⵽����Ӱ��������"<<chromaticityShadowNum<<endl;

	/*	//����struct�洢����Ӱ��Ϣ�Ƿ���ɫ�Ȳ�����һ��
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
	for(int j=0;j<WIDTH;j++)
	{	
	if(graph[i][j].category==2)
	testNum++;
	}
	}
	cout<<"�ṹ���д洢����Ӱ��Ϊ:"<<testNum<<endl;

	cout<<"row="<<sceneMat.rows<<endl;
	cout<<"col="<<sceneMat.cols<<endl;
	*/

	namedWindow("ɫ�Ȳ�����",WINDOW_NORMAL);
	imshow("ɫ�Ȳ�����", chromaticityMat);
	waitKey(0);
	destroyWindow("ǰ��ͼ");
	destroyWindow("ɫ�Ȳ�����");

	//����ͼƬ
	//����Ϊbmp��ʽ��ͼƬ������ѹ��������Ϊjpg��ʽ��ͼƬ����ѹ������ʹ��CV_IMWRITE_JPEG_QUALITY����Ϊ100Ҳ����
	vector<int>imwriteJPGquality;
	imwriteJPGquality.push_back(CV_IMWRITE_JPEG_QUALITY);   //JPG��ʽͼƬ������
	imwriteJPGquality.push_back(100);
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\201702281110043_chromaticity.jpg", chromaticityMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\201702281110043_chromaticity.bmp", chromaticityMat);
	return 0;
}


//����ǰ��ͼ���뱳��ͼ������Ȳ�
//�����ǰ��ͼƬ�Ѿ�����Ϊ�����ǻ�ɫ����Ӱ��ɫ�������ɫ
int brightnessDiffer()
{
	//�������ģ�ֻ��ɫ�Ȳ��⵽����Ӱ����ɸѡ
	//��ÿ�����ص�BDֵ���浽txt�ļ���
	//B����
	ofstream out_bdB("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_B.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdB<<bd_B[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_bdB<<endl;   //ÿ�������������ӻ���
	}
	//G����
	ofstream out_bdG("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_G.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdG<<bd_G[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_bdG<<endl;   //ÿ�������������ӻ���
	}
	//R����
	ofstream out_bdR("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_R.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdR<<bd_R[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_bdR<<endl;   //ÿ�������������ӻ���
	}
	out_bdB.close();
	out_bdG.close();
	out_bdR.close();

	//����BD������
	bd_m_B=bd_m_B/chromaticityShadowNum;
	bd_m_G=bd_m_G/chromaticityShadowNum;
	bd_m_R=bd_m_R/chromaticityShadowNum;

	//����BD�ķ���
	bd_variance_B=0;
	bd_variance_G=0;
	bd_variance_R=0;
	for(int i=0;i<sceneMat.rows; i++)
	{
		for(int j=0; j<sceneMat.cols; j++)
		{
			if(chromaticityMat.at<Vec3b>(i,j)[0]==0 && chromaticityMat.at<Vec3b>(i,j)[1]==255 && chromaticityMat.at<Vec3b>(i,j)[0]==0)
			{
				bd_variance_B=bd_variance_B + pow((bd_B[i][j]-bd_m_B),2);  //B����
				bd_variance_G=bd_variance_G + pow((bd_G[i][j]-bd_m_G),2);  //G����
				bd_variance_R=bd_variance_R + pow((bd_R[i][j]-bd_m_R),2);  //R����
			}
		}
	}
	bd_variance_B=sqrt(bd_variance_B/chromaticityShadowNum);
	bd_variance_G=sqrt(bd_variance_G/chromaticityShadowNum);
	bd_variance_R=sqrt(bd_variance_R/chromaticityShadowNum);

	//����RGB���������ĸߵ���ֵ
	bd_thresholdH_B= bd_m_B + 1.96*bd_variance_B;  //B����
	bd_thresholdL_B= bd_m_B - 1.96*bd_variance_B;
	bd_thresholdH_G= bd_m_G + 1.96*bd_variance_G;  //G����
	bd_thresholdL_G= bd_m_G - 1.96*bd_variance_G;
	bd_thresholdH_R= bd_m_R + 1.96*bd_variance_R;  //R����
	bd_thresholdL_R= bd_m_R - 1.96*bd_variance_R;

	cout<<endl;
	cout<<"--------------------BD������----------------------"<<endl;
	cout<<"BD_B��������"<<bd_m_B<<endl;
	cout<<"BD_G��������"<<bd_m_G<<endl;
	cout<<"BD_R��������"<<bd_m_R<<endl;
	cout<<"BD_B�ķ��"<<bd_variance_B<<endl;
	cout<<"BD_G�ķ��"<<bd_variance_G<<endl;
	cout<<"BD_R�ķ��"<<bd_variance_R<<endl;
	cout<<"B�ķ�����ֵ��"<<bd_thresholdL_B<<"\t"<<bd_thresholdH_B<<endl;
	cout<<"G�ķ�����ֵ��"<<bd_thresholdL_G<<"\t"<<bd_thresholdH_G<<endl;
	cout<<"R�ķ�����ֵ��"<<bd_thresholdL_R<<"\t"<<bd_thresholdH_R<<endl;

	brightnessMat=chromaticityMat.clone();   //�����brightnessMat������chromaticityMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	namedWindow("�Աȣ�ɫ�Ȳ�����",WINDOW_NORMAL);
	imshow("�Աȣ�ɫ�Ȳ�����", chromaticityMat);
	waitKey(0);
	//ɫ�Ȳ�����������Ϊ��ɫ�������ɫ����Ӱ��ɫ�����Ȳ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
	int addObject=0;   //���ȱ��¼���������������
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			//���Ȳ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//BD����ֵ�����ڣ�������Ӱ
				//if( (bd_B[i][j]>bd_thresholdL_B && bd_B[i][j]<bd_thresholdH_B) || (bd_G[i][j]>bd_thresholdL_G && bd_G[i][j]<bd_thresholdH_G) || (bd_R[i][j]>bd_thresholdL_R && bd_R[i][j]<bd_thresholdH_R) )
				if( (bd_B[i][j]>bd_thresholdL_B && bd_B[i][j]<bd_thresholdH_B) && (bd_G[i][j]>bd_thresholdL_G && bd_G[i][j]<bd_thresholdH_G) && (bd_R[i][j]>bd_thresholdL_R && bd_R[i][j]<bd_thresholdH_R) )
				{
					continue;
				}  
				else
				{
					brightnessMat.at<Vec3b>(i,j)[0]=255;   //�¼�������Ϊ��ɫ
					brightnessMat.at<Vec3b>(i,j)[1]=0;
					brightnessMat.at<Vec3b>(i,j)[2]=0;

					addObject++;  //���ȱ��¼���������������

					//�޸Ľṹ����Ӧ��Ϣ
					graph[i][j].category=1;
					graph[i][j].initColor_B=0;
					graph[i][j].initColor_G=0;
					graph[i][j].initColor_R=255;
				}
			}
		}
	}
	cout<<"���ȱ��¼�����������������"<<addObject<<endl;
	//����struct�洢����Ӱ��Ϣ�Ƿ���ɫ�Ȳ�����һ��
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{	
			if(graph[i][j].category==2)
				testNum++;
		}
	}
	cout<<"��ǰ��Ӱ��������Ϊ:"<<testNum<<endl;

	//�˲���ʾ��Ϊ�˽����Ȳ�������ɫ�Ȳ��������жԱ�
	namedWindow("�Աȣ����Ȳ�����",WINDOW_NORMAL);
	imshow("�Աȣ����Ȳ�����", brightnessMat);
	waitKey(0);
	destroyWindow("�Աȣ�ɫ�Ȳ�����");
	destroyWindow("�Աȣ����Ȳ�����");

	//����Ա�ͼƬ
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.jpg", brightnessMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.bmp", brightnessMat);

	//����ɫ��+���ȵĽ����ͳһ��ɫ����Ӱ��ɫ�������ɫ
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			if( abs(brightnessMat.at<Vec3b>(i,j)[0]-255)==0 && abs(brightnessMat.at<Vec3b>(i,j)[1]-0)==0 && abs(brightnessMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				brightnessMat.at<Vec3b>(i,j)[0]=0;   	//���ϴ��¼�⵽��������ɫ����ɫ��Ϊ��ɫ����
				brightnessMat.at<Vec3b>(i,j)[1]=0;
				brightnessMat.at<Vec3b>(i,j)[2]=255;
			}
		}
	}
	namedWindow("ɫ��+���Ȳ�����",WINDOW_NORMAL);
	imshow("ɫ��+���Ȳ�����", brightnessMat);
	waitKey(0);
	destroyWindow("ɫ��+���Ȳ�����");
	//�����һ������ͼƬ
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.jpg", brightnessMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.bmp", brightnessMat);

	return 0;
}

//�ֲ����ȱ�
int localRelation()
{
	cout<<"-------------�ֲ�ǿ�ȱȼ����Ӱ------------------"<<endl;
	//	localMat=brightnessMat.clone();   //�����localMat������brightnessMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	localMat=imread("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.bmp");  //��ȡͼ��
	namedWindow("ɫ��+���Ȳ�����",WINDOW_NORMAL);
	imshow("ɫ��+���Ȳ�����", localMat);
	waitKey(0);

	//��ʼ��Qֵ
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			q_B[i][j]=0;
			q_G[i][j]=0;
			q_R[i][j]=0;
		}
	}

	//ɫ�Ȳ�+���Ȳ�����������Ϊ��ɫ�������ɫ����Ӱ��ɫ���ֲ����ȱ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
	//int sNum=0, notBoarder=0;
	for(int i=1;i<localMat.rows-1;i++)  //ע��ͼƬ��Ե�������Qֵ��ע��i��j��ȡֵ��Χ
	{
		for(int j=1;j<localMat.cols-1;j++)
		{
			//�ֲ����ȱ�ֻ������Ӱ��ѡ������ɫ��������ĳ���������廹����Ӱ
			if( abs(localMat.at<Vec3b>(i,j)[0]-0)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//sNum++;  //��Ӱ���ظ���

				//�ų���Ӱ��Ե: ���ص�����ҲҪ������Ӱ
				if ( (localMat.at<Vec3b>(i,j-1)[0]==0 && localMat.at<Vec3b>(i,j-1)[1]==255 && localMat.at<Vec3b>(i,j-1)[2]==0) && (localMat.at<Vec3b>(i+1,j)[0]==0 && localMat.at<Vec3b>(i+1,j)[1]==255 && localMat.at<Vec3b>(i+1,j)[2]==0) && (localMat.at<Vec3b>(i,j+1)[0]==0 && localMat.at<Vec3b>(i,j+1)[1]==255 && localMat.at<Vec3b>(i,j+1)[2]==0) && (localMat.at<Vec3b>(i-1,j)[0]==0 && localMat.at<Vec3b>(i-1,j)[1]==255 && localMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					q_B[i][j]= pow((bd_B[i][j-1]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i+1][j]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i][j+1]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i-1][j]-bd_m_B)/bd_variance_B,2);
					q_G[i][j]= pow((bd_G[i][j-1]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i+1][j]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i][j+1]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i-1][j]-bd_m_G)/bd_variance_G,2);
					q_R[i][j]= pow((bd_B[i][j-1]-bd_m_R)/bd_variance_R,2)+ pow((bd_B[i+1][j]-bd_m_R)/bd_variance_R,2)+ pow((bd_R[i][j+1]-bd_m_R)/bd_variance_R,2)+ pow((bd_B[i-1][j]-bd_m_R)/bd_variance_R,2);

					/*//���Qֵ
					cout<<"q_B="<<q_B[i][j]<<"\t"<<"q_G="<<q_G[i][j]<<"\t"<<"q_R="<<q_R[i][j]<<endl;
					notBoarder++;  //�Ǳ�Ե��Ӱ���ظ���
					*/
				}
			}
		}
	}
	//cout<<"shadow Num:"<<sNum<<endl;
	//cout<<"shadow exclude boarder Num:"<<notBoarder<<endl;

	//����Qֵ����ɫ��+���ȼ�⵽����Ӱ���ٴ����ж�
	int addObject=0;   //�ֲ���ϵ�¼���������������
	for(int i=1;i<localMat.rows-1;i++)  //ע��ͼƬ��Ե�������Qֵ��ע��i��j��ȡֵ��Χ
	{
		for(int j=1;j<localMat.cols-1;j++)
		{
			//�ֲ����ȱ�ֻ������Ӱ��ѡ������ɫ��������ĳ���������廹����Ӱ
			if( abs(localMat.at<Vec3b>(i,j)[0]-0)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//�ų���Ӱ��Ե: ���ص�����ҲҪ������Ӱ
				if ( (localMat.at<Vec3b>(i,j-1)[0]==0 && localMat.at<Vec3b>(i,j-1)[1]==255 && localMat.at<Vec3b>(i,j-1)[2]==0) && (localMat.at<Vec3b>(i+1,j)[0]==0 && localMat.at<Vec3b>(i+1,j)[1]==255 && localMat.at<Vec3b>(i+1,j)[2]==0) && (localMat.at<Vec3b>(i,j+1)[0]==0 && localMat.at<Vec3b>(i,j+1)[1]==255 && localMat.at<Vec3b>(i,j+1)[2]==0) && (localMat.at<Vec3b>(i-1,j)[0]==0 && localMat.at<Vec3b>(i-1,j)[1]==255 && localMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					//if(q_B[i][j]<9.49 || q_G[i][j]<9.49 || q_R[i][j]<9.49) //��Ӱ 
					if(q_B[i][j]<9.49 && q_G[i][j]<9.49 && q_R[i][j]<9.49) //��Ӱ 
						continue;   
					else
					{
						localMat.at<Vec3b>(i,j)[0]=255;   //�¼�������Ϊ��ɫ
						localMat.at<Vec3b>(i,j)[1]=255;
						localMat.at<Vec3b>(i,j)[2]=255;

						addObject++;  //�ֲ���ϵ�¼���������������

						//�޸Ľṹ����Ӧ��Ϣ
						graph[i][j].category=1;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=0;
						graph[i][j].initColor_R=255;
					}
				}
			}
		}
	}
	cout<<"�ֲ���ϵ�¼�����������������"<<addObject<<endl;
	//����struct�洢����Ӱ��Ϣ�Ƿ���ɫ�Ȳ�����һ��
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{	
			if(graph[i][j].category==2)
				testNum++;
		}
	}
	cout<<"��ǰ��Ӱ��������Ϊ:"<<testNum<<endl;

	//�˲���ʾ��Ϊ�˽��ֲ��Աȼ������ɫ��+���Ȳ��������жԱ�
	namedWindow("�ֲ��Աȼ����",WINDOW_NORMAL);
	imshow("�ֲ��Աȼ����", localMat);
	waitKey(0);
	destroyWindow("�ֲ��Աȼ����");
	destroyWindow("ɫ��+���Ȳ�����");

	//����Ա�ͼƬ
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.jpg", localMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.bmp", localMat);

	//����ɫ��+����+�ֲ��ԱȵĽ����ͳһ��ɫ����Ӱ��ɫ�������ɫ
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			if( abs(localMat.at<Vec3b>(i,j)[0]-255)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-255)==0 )
			{
				localMat.at<Vec3b>(i,j)[0]=0;   	//���ϴ��¼�⵽��������ɫ�ɰ�ɫ��Ϊ��ɫ����
				localMat.at<Vec3b>(i,j)[1]=0;
				localMat.at<Vec3b>(i,j)[2]=255;
			}
		}
	}
	namedWindow("ɫ��+���Ȳ�+�ֲ��Աȼ����",WINDOW_NORMAL);
	imshow("ɫ��+���Ȳ�+�ֲ��Աȼ�����", localMat);
	waitKey(0);
	destroyWindow("ɫ��+���Ȳ�+�ֲ��Աȼ�����");
	//�����һ������ͼƬ
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.jpg", localMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp", localMat);

	return 0;
}

//������ͨ��İ�Χ��ϵ�Ż���Ӱ������
int spatialAjustment()
{
	cout<<"-------------������ͨ��İ�Χ��ϵ�Ż����------------------"<<endl;
	//spacialMat=localMat.clone();   //�����spacialMat������localMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp");  //��ȡͼ��
	//spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Color Space\\RGB\\Img_Rgb.jpg");
	namedWindow("�Աȣ�ɫ��+���Ȳ�+�ֲ������",WINDOW_NORMAL);
	imshow("�Աȣ�ɫ��+���Ȳ�+�ֲ������", spacialMat);
	waitKey(0);

	//��ԭͼתΪ�Ҷ�ͼ
	cvtColor(spacialMat, spacialGrayMat, CV_BGR2GRAY);

	//����ԭͼ���ڲ���ʾ
	namedWindow(WINDOW_NAME1, CV_WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, spacialMat);
	waitKey(0);


	//������������������ֵ
	/*  ��һ������������������
	�ڶ�����������������
	�����������������������ϵ�ʱ��opencv���Զ�����ǰλ���������ֵ���ݸ�ָ��ָ�������
	���ĸ����������������ܵ�������ֵ
	�������������ѡ�Ļص�����,����Ϊ�Զ������ֵ����
	*/
	createTrackbar("��ֵ", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange);
	on_ThreshChange(0,0);   //��ʼ���Զ������ֵ����

	//�ȴ��û������������ESC�����˳��ȴ�����
	while (true)
	{
		int c;
		c=waitKey(20);
		if((char)c==27)
			break;
	}

	return 0;
}

//�Զ������ֵ����
void on_ThreshChange(int, void*)
{
	Mat src_copy=spacialMat.clone();
	Mat threshold_output;
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//��ͼ����ж�ֵ��
	/*  threshold�����������Ҷ�ͼ����ͼ����Ϣ��ֵ������������ͼƬֻ������ɫֵ
	��һ�����������룬����Ϊ��ͨ����8bit��32bit�������͵�Mat����
	�ڶ�������������������������һ����������ͬ�ĳߴ������
	��������������ֵ�ľ���ֵ
	���ĸ�������maxvalue�������������ȡTHRESH_BINARY��THRESH_BINARY_INV����ʱ�����ֵ����ֵ����0�ڣ�255�ף�
	�������������ֵ���ͣ�THRESH_BINARY ��ǰ�������ֵʱ��ȡmaxvalue�������ĸ�����������������Ϊ0
	*/
	threshold(spacialGrayMat, threshold_output, g_nThresh, 255, THRESH_BINARY);

	
	//Ѱ������
	/*  ��һ������������ͼ��8bit�ĵ�ͨ����ֵͼ��
	contours����⵽����������һ��������ÿ��Ԫ�ض���һ����������ˣ����������ÿ��Ԫ�ض���һ����������vector<vector<Point>>contours
	hierarchy:���������ļ̳й�ϵ��hierarchyҲ��һ��������������contours��ȣ�ÿ��Ԫ�غ�contours��Ԫ�ض�Ӧ��
	hierarchy��ÿ��Ԫ����һ�������ĸ�����������������vector<Vec4i>hierarchy
	hierarchy[i][0],hierarchy[i][1],hierarchy[i][2],hierarchy[i][3]�ֱ��ʾ��i��������contours[i])����һ����ǰһ���������ĵ�һ���������Ͱ������ĸ�����
	���ĸ���������������ķ������������֡�CV_RETR_TREE����������������������еļ̳У���������ϵ��
	�������������ʾһ�������ķ�����CV_CHAIN_APPROX_SIMPLEֻ�洢ˮƽ����ֱ���Խ�ֱ�ߵ���ʼ�㡣
	������������ÿһ���������ƫ�������������Ǵ�ͼ��ROI�У�����Ȥ������ȡ������ʱ��ʹ��ƫ�������ã���Ϊ���Դ�����ͼ����������������������
	���磬���ͼ���(100,0)��ʼ����������⣬�ʹ��루100��0��
	*/
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	//��ÿ������������͹��
	//͹����һ��������������������Χ������ǣ����б������С����ǣ�����͹��
	vector<vector<Point>>hull(contours.size());
	for(int i=0; i<contours.size(); i++)
	{
		/*  ��һ��������Ҫ���͹���ĵ㼯
		�ڶ��������������͹����
		������������bool��������ʾ��õ�͹����˳ʱ�뻹����ʱ�뷽��true��˳ʱ��
		*/
		convexHull(Mat(contours[i]), hull[i], false);
	}

	//�����������͹��
	Mat drawing=Mat::zeros(threshold_output.size(), CV_8UC3);   //����ָ����С�����͵�������
	for(int i=0; i<contours.size(); i++)
	{
		//scalar������ɴ��1--4����ֵ������
		//uniform������ָ����Χ�������
		Scalar color=Scalar(g_rng.uniform(0,255), g_rng.uniform(0,255), g_rng.uniform(0,255));

		/*drawContours������ͼ�������
		��һ��������Ŀ��ͼ��
		�ڶ�������������������飬ÿһ�������ɵ�vector����
		������������ָ�����ڼ�������
		���ĸ���������������ɫ
		������������������߿����Ϊ��ֵ����CV_FILLED��ʾ��������ڲ�
		����������������������
		���߸������������ṹ��Ϣ
		�ڰ˸�������MAX_LEVEL���������������ȼ������Ϊ0�����Ƶ��������������Ϊ1������������������ͬ��������������Ϊ2�����е�������
		�ھŸ����������ո�����ƫ�����ƶ�ÿһ�����������ꡣ�������Ǵ�ĳЩ����Ȥ����ROI������ȡʱ����Ҫ����ROIƫ���������õ��������
		*/
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		/*		//����ÿ�����������
		double area = fabs(contourArea(contours[i], true));
		cout<<"��"<<i<<"�����������Ϊ��"<<area<<endl;
		*/
	}

	//�ѽ����ʾ�ڴ���
	namedWindow(WINDOW_NAME2, CV_WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME2, drawing);
}

//���С��ͨ��
void fillSmallDomain()
{
	spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp");  //��ȡͼ��
	//ͳ�Ƶ�ǰ������(��ɫ�����ظ���
	int objectNum=0;
	int shadowNum=0;
	int backNum=0;
	for(int i=0;i<spacialMat.rows;i++)
	{
		for(int j=0;j<spacialMat.cols;j++)
		{
			graph[i][j].revise=0;
			//���壺��ɫ
			//if( abs(spacialMat.at<Vec3b>(i,j)[0]-0)==0 && abs(spacialMat.at<Vec3b>(i,j)[1]-0)==0 && abs(spacialMat.at<Vec3b>(i,j)[2]-255)==0 )
			if( spacialMat.at<Vec3b>(i,j)[0]==0 && spacialMat.at<Vec3b>(i,j)[1]==0 && spacialMat.at<Vec3b>(i,j)[2]==255 )	
			{
				objectNum++;
				graph[i][j].category=1;
			}
			else if( spacialMat.at<Vec3b>(i,j)[0]==0 && spacialMat.at<Vec3b>(i,j)[1]==255 && spacialMat.at<Vec3b>(i,j)[2]==0 )
			{
				shadowNum++;
				graph[i][j].category=2;
			}
			else
			{
				backNum++;
				graph[i][j].category=0;
			}
		}
	}
	cout<<"��ǰ�������ظ�����"<<objectNum<<endl;
	cout<<"��ǰ��Ӱ���ظ�����"<<shadowNum<<endl;
	cout<<"��ǰ�������ظ�����"<<backNum<<endl;


	//����С��ͨ����������������4%
	double connectedDomain;
	connectedDomain = objectNum * 0.04;
	cout<<"��ͨ����������ظ�����"<<connectedDomain<<endl;

	//-------------����֮ǰ����ֵ�����������������ŵ���ֵ------------------------------
	IplImage* src=NULL;
	IplImage* img=NULL;
	IplImage* dst=NULL;

	CvMemStorage* storage=cvCreateMemStorage(0);
	CvSeq* contour=0;

	CvScalar external_color;  //��������ɫ��ͼ���ֵ����ֻ�к�ɫ�Ͱ�ɫ����ɫ����������ǡ���������
	CvScalar hole_color;  //��������ɫ����ɫ����������ǡ���������

	src=cvLoadImage("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp",1);
	img=cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	dst=cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

	cvCvtColor(src, img, CV_BGR2GRAY);
	//ע�⣡���������˴���100�ǻ���֮ǰ�ֶ���ֵ�����Լ����õ�����Ѱ����������ֵ��������
	cvThreshold(img, img, 100, 200, CV_THRESH_BINARY);

	//�ҵ���ֵͼ���е�����
	/*  CV_RETR_LIST����ȡ������������������list��
	CV_CHAIN_APPROX_NONE�������е���������ʽת��Ϊ��������ʽ
	*/
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cvZero(dst);  //�������

	//_contour��Ϊ�˱�����������ָ��λ�ã���Ϊ���contour����������
	CvSeq* _contour=contour;

	//----------------------����������������-------------
	IplImage* test=NULL;
	test=cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	CvScalar colorEx1=CV_RGB(255, 0, 0);  //���������ɫ
	CvScalar colorEx2=CV_RGB(0, 255, 0);  //�ڲ�������ɫ

	CvSeq* cNext=NULL;
	bool first=true;
	int count=0;  //��������
	int tempRow,tempCol;
	int tempB,tempG,tempR;
	int up=0,right=0,left=0,down=0,none=0,upleft=0,upright=0,downleft=0,downright=0;  //������
	for(CvSeq* c=contour; c!=NULL; c=cNext)
	{
		double tmp=fabs(cvContourArea(c));  //�����������
		//------------ɾ������ͨ��------------
		if(tmp>connectedDomain)  //���������С��ͨ��ɾ��
		{
			//ɾ�������
			cNext=c->h_next;
			cvClearSeq(c);
			continue;
		}
		else
		{
			if(first)  //��������еĵ�һ��������ɾ������������ָ��ָ��������һ��Ԫ��
				contour=c;
			first=false;

			//**************���С����*****************
			//c->total������contours_tmp�е������
			for(int j=0; j<c->total; j++)  //��ȡһ�����������������
			{
				//cvGetSeqElem��������������ָ����Ԫ��ָ��
				CvPoint *pt=(CvPoint*)cvGetSeqElem(c, j);  //cvGetSeqElem���õ������е�һ����

				//�洢��λ�����ص�λ����Ϣ
				tempRow=pt->y;
				tempCol=pt->x;
				//�ų�ͼ���Ե
				if(tempRow!=0 || tempRow!=HEIGHT-1 || tempCol!=0 || tempCol!=WIDTH-1)
				{
					//�洢��λ�����ص���ɫ��Ϣ
					tempB=spacialMat.at<Vec3b>(tempRow,tempCol)[0];
					tempG=spacialMat.at<Vec3b>(tempRow,tempCol)[1];
					tempR=spacialMat.at<Vec3b>(tempRow,tempCol)[2];

					//�����ذ��������ɫ��Ϣ
					//�������������λ����ɫ��ͬʱ���Ͳ��ؼ����������������أ�����ֱ�ӶԸ������������
					int b,g,r;
					//�ϱߵĵ�
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						// cvDrawContours()����ͼ���ϻ����ⲿ�ڲ�����
						//��һ��������Ҫ�����ϻ���������ͼ��
						//�ڶ���������ָ���һ��������ָ��
						//����������������������ɫ
						//���ĸ�����������������ɫ
						//���������������������������0��ֻ����contours_tmp
						//���������������������ߵĿ�ȡ�CV_FILLED��contours_tmp�ڲ���������
						//���߸������������߶ε�����
						//�ڰ˸�������������ֵ�ƶ����е������
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						up++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ
							
						break;
					}

					//�ұߵĵ�
					b=spacialMat.at<Vec3b>(tempRow,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						right++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//�±ߵĵ�
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						down++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//��ߵĵ�
					b=spacialMat.at<Vec3b>(tempRow,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						left++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//���ϱߵĵ�
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						upleft++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//���ϱߵĵ�
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						upright++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//���±ߵĵ�
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						downleft++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					//���±ߵĵ�
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						downright++;

						//�޸����ص���Ϣ
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //����
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //����
						}
						else
							graph[tempRow][tempCol].category=2;  //��Ӱ

						break;
					}

					none++;
					cvDrawContours(test, c, CV_RGB(tempR,tempG,tempB), CV_RGB(tempR,tempG,tempB), 0, CV_FILLED, 8, cvPoint(0,0));
					continue;
				}
				
			}

			count++;
		}
		cNext=c->h_next;
	}
	cout<<"С��ͨ�������"<<count<<endl;
	cout<<"up������"<<up<<endl;
	cout<<"right������"<<right<<endl;
	cout<<"down������"<<down<<endl;
	cout<<"left������"<<left<<endl;
	cout<<"upleft������"<<upleft<<endl;
	cout<<"upright������"<<upright<<endl;
	cout<<"downleft������"<<downleft<<endl;
	cout<<"downright������"<<downright<<endl;
	cout<<"none������"<<none<<endl;
	cvNamedWindow("fill image", CV_WINDOW_AUTOSIZE);
	cvShowImage("fill image", test);
	cvWaitKey(0);


	//------------�����µ�ͼ��-------
	Mat testMat(test, 0);
	for(int i=0; i<testMat.rows; i++)
	{
		for(int j=0; j<testMat.cols; j++)
		{
			if( (testMat.at<Vec3b>(i,j)[0]==0 && testMat.at<Vec3b>(i,j)[1]==255 && testMat.at<Vec3b>(i,j)[2]==255) || (testMat.at<Vec3b>(i,j)[0]==0 && testMat.at<Vec3b>(i,j)[1]==0 && testMat.at<Vec3b>(i,j)[2]==255) || (testMat.at<Vec3b>(i,j)[0]==0 && testMat.at<Vec3b>(i,j)[1]==255 && testMat.at<Vec3b>(i,j)[2]==0))
				graph[i][j].revise=1;
		}
	}

	for(int i=0;i<spacialMat.rows;i++)
	{
		for(int j=0;j<spacialMat.cols;j++)
		{
			if(graph[i][j].revise==1)
			{
				spacialMat.at<Vec3b>(i,j)[0]=testMat.at<Vec3b>(i,j)[0];   
				spacialMat.at<Vec3b>(i,j)[1]=testMat.at<Vec3b>(i,j)[1];
				spacialMat.at<Vec3b>(i,j)[2]=testMat.at<Vec3b>(i,j)[2];
			}
		}
	}
	namedWindow("�����С��ͨ��",WINDOW_NORMAL);
	imshow("�����С��ͨ��", spacialMat);
	waitKey(0);
	destroyWindow("�����С��ͨ��");
	cvDestroyWindow("filter image");	
	cvDestroyWindow("fill image");
	cvReleaseImage(&test);
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvReleaseMemStorage(&storage);

	//�����һ������ͼƬ
	imwrite("F:\\Code\\Shadow Detection\\Data\\Spacial Improved\\20170228111043_brightness+local+spacial.bmp", spacialMat);
}


//��Ӱ����㷨
int shadowDetection()
{
	//step1. ɫ�Ȳ���Ӱ���
//	chromaticityDiffer();

	//step2. ���Ȳ���Ӱ���
	//ע���������Ҫ�õ�chromaticityDiffer()
//	brightnessDiffer();

	//step3. �ֲ����ȱ�
//	localRelation();

	//step4.������ͨ��İ�Χ��ϵ�Ż���Ӱ������
//	spatialAjustment();  //�ֶ�ѡȡ��ֵ

	//step5.�����С��ͨ��
	fillSmallDomain();   //�����С��ͨ��

	return 0;
}