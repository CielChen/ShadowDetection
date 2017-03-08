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

Mat sceneMat, backgroundMat;
Mat chromaticityMat, brightnessMat, localMat, spacialMat;  //�洢ÿ����Ӱ�����

int sceneRGB_B[WIDTH][HEIGHT],sceneRGB_G[WIDTH][HEIGHT],sceneRGB_R[WIDTH][HEIGHT];  //ǰ��ͼ���RGB����
int backgroundRGB_B[WIDTH][HEIGHT],backgroundRGB_G[WIDTH][HEIGHT],backgroundRGB_R[WIDTH][HEIGHT];  //����ͼ���RGB����
double sceneNorm[WIDTH][HEIGHT],backgroundNorm[WIDTH][HEIGHT];  //ǰ��ͼ��ͱ���ͼ��ÿ������RGB������2����
double cd_B[WIDTH][HEIGHT],cd_G[WIDTH][HEIGHT],cd_R[WIDTH][HEIGHT];  //ɫ�Ȳ��RGB����
double bd_B[WIDTH][HEIGHT],bd_G[WIDTH][HEIGHT],bd_R[WIDTH][HEIGHT];  //���Ȳ��RGB����
double q_B[WIDTH][HEIGHT], q_G[WIDTH][HEIGHT], q_R[WIDTH][HEIGHT]; //RGB����������Qֵ

double m_B, m_G, m_R;  //RGB��������������
double variance_B, variance_G, variance_R;   //RGB���������ķ���
double thresholdH_B, thresholdL_B, thresholdH_G, thresholdL_G, thresholdH_R, thresholdL_R;  //RGB���������ĸߵ���ֵ

int chromaticityShadowNum;  //ɫ�Ȳ��⵽����Ӱ������

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
	sceneMat=imread("G:\\Code-Shadow Detection\\Data\\Foreground\\20170228111043_fore.jpg");  //�ʼ��ȡ����ǰ��ͼ��
	backgroundMat=imread("G:\\Code-Shadow Detection\\Data\\Background\\20170228111043_back.jpg");  //����ͼ��

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
	ofstream out_cdB("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_B.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdB<<cd_B[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_cdB<<endl;   //ÿ�������������ӻ���
	}
	//G����
	ofstream out_cdG("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_G.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdG<<cd_G[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_cdG<<endl;   //ÿ�������������ӻ���
	}
	//R����
	ofstream out_cdR("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_R.txt");  //���ļ�
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
	m_B=0;
	m_G=0;
	m_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_B++;
				m_B=m_B+cd_B[i][j];
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_G++;
				m_G=m_G+cd_G[i][j];
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R������������������Ϊ[-0.2,0.2]��CD
			{
				cdNum_R++;
				m_R=m_R+cd_R[i][j];
			}
		}
	}
	m_B=m_B/cdNum_B;
	m_G=m_G/cdNum_G;
	m_R=m_R/cdNum_R;

	//����CD�ķ���
	variance_B=0;
	variance_G=0;
	variance_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B������������������Ϊ[-0.2,0.2]��CD
			{
				variance_B=variance_B + pow((cd_B[i][j]-m_B),2);
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G������������������Ϊ[-0.2,0.2]��CD
			{
				variance_G=variance_G + pow((cd_G[i][j]-m_G),2);
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R������������������Ϊ[-0.2,0.2]��CD
			{
				variance_R=variance_R + pow((cd_R[i][j]-m_R),2);
			}
		}
	}
	variance_B=sqrt(variance_B/cdNum_B);
	variance_G=sqrt(variance_G/cdNum_G);
	variance_R=sqrt(variance_R/cdNum_R);

	//����RGB���������ĸߵ���ֵ
	thresholdH_B= m_B + 1.96*variance_B;  //B����
	thresholdL_B= m_B - 1.96*variance_B;
	thresholdH_G= m_G + 1.96*variance_G;  //G����
	thresholdL_G= m_G - 1.96*variance_G;
	thresholdH_R= m_R + 1.96*variance_R;  //R����
	thresholdL_R= m_R - 1.96*variance_R;

	cout<<"--------------------CD������----------------------"<<endl;
	cout<<"[-0.2,0.2]cdNum_B��"<<cdNum_B<<endl;
	cout<<"[-0.2,0.2]cdNum_G��"<<cdNum_G<<endl;
	cout<<"[-0.2,0.2]cdNum_R��"<<cdNum_R<<endl;
	cout<<"CD_B��������"<<m_B<<endl;
	cout<<"CD_G��������"<<m_G<<endl;
	cout<<"CD_R��������"<<m_R<<endl;
	cout<<"CD_B�ķ��"<<variance_B<<endl;
	cout<<"CD_G�ķ��"<<variance_G<<endl;
	cout<<"CD_R�ķ��"<<variance_R<<endl;
	cout<<"B�ķ�����ֵ��"<<thresholdL_B<<"\t"<<thresholdH_B<<endl;
	cout<<"G�ķ�����ֵ��"<<thresholdL_G<<"\t"<<thresholdH_G<<endl;
	cout<<"R�ķ�����ֵ��"<<thresholdL_R<<"\t"<<thresholdH_R<<endl;

	chromaticityMat=sceneMat.clone();   //�����chromaticityMat������sceneMat���γ�һ���µ�ͼ����������໥û��Ӱ��	
	chromaticityShadowNum=0;  //ɫ�Ȳ��⵽����Ӱ������
	//����BD������
	m_B=0;
	m_G=0;
	m_R=0;
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
				}
				else
				{
					//CD����ֵ�����ڣ�������Ӱ
					if( (cd_B[i][j]>thresholdL_B && cd_B[i][j]<thresholdH_B) || (cd_G[i][j]>thresholdL_G && cd_G[i][j]<thresholdH_G) || (cd_R[i][j]>thresholdL_R && cd_R[i][j]<thresholdH_R) )
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;  //��ӰΪ��ɫ
						chromaticityMat.at<Vec3b>(i,j)[1]=255;
						chromaticityMat.at<Vec3b>(i,j)[2]=0;

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
						m_B=m_B+bd_B[i][j];  //B����
						m_G=m_G+bd_G[i][j];  //G����
						m_R=m_R+bd_R[i][j];  //R����
					}  
					else
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;   //����Ϊ��ɫ
						chromaticityMat.at<Vec3b>(i,j)[1]=0;
						chromaticityMat.at<Vec3b>(i,j)[2]=255;
					}

				}
			}

		}
	}
	cout<<"ɫ�Ȳ��⵽����Ӱ��������"<<chromaticityShadowNum<<endl;

	namedWindow("ɫ�Ȳ�����",WINDOW_NORMAL);
	imshow("ɫ�Ȳ�����", chromaticityMat);
	waitKey(0);
	destroyWindow("ǰ��ͼ");
	destroyWindow("ɫ�Ȳ�����");

	//����ͼƬ
	//	imwrite("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\example_chromaticity.jpg", result);
	imwrite("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\20170228111043_chromaticity.jpg", chromaticityMat);

	return 0;
}


//����ǰ��ͼ���뱳��ͼ������Ȳ�
//�����ǰ��ͼƬ�Ѿ�����Ϊ�����ǻ�ɫ����Ӱ��ɫ�������ɫ
int brightnessDiffer()
{
	//�������ģ�ֻ��ɫ�Ȳ��⵽����Ӱ����ɸѡ
	//��ÿ�����ص�BDֵ���浽txt�ļ���
	//B����
	ofstream out_bdB("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_B.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdB<<bd_B[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_bdB<<endl;   //ÿ�������������ӻ���
	}
	//G����
	ofstream out_bdG("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_G.txt");  //���ļ�
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdG<<bd_G[i][j]<<"\t";  //��ÿ��Ԫ��д���ļ�����Tab�ָ�   ע��������������-1.#IND���ʾ��С����ȷ��
		}
		out_bdG<<endl;   //ÿ�������������ӻ���
	}
	//R����
	ofstream out_bdR("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_R.txt");  //���ļ�
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
	m_B=m_B/chromaticityShadowNum;
	m_G=m_G/chromaticityShadowNum;
	m_R=m_R/chromaticityShadowNum;

	//����BD�ķ���
	variance_B=0;
	variance_G=0;
	variance_R=0;
	for(int i=0;i<sceneMat.rows; i++)
	{
		for(int j=0; j<sceneMat.cols; j++)
		{
			if(chromaticityMat.at<Vec3b>(i,j)[0]==0 && chromaticityMat.at<Vec3b>(i,j)[1]==255 && chromaticityMat.at<Vec3b>(i,j)[0]==0)
			{
				variance_B=variance_B + pow((bd_B[i][j]-m_B),2);  //B����
				variance_G=variance_G + pow((bd_G[i][j]-m_G),2);  //G����
				variance_R=variance_R + pow((bd_R[i][j]-m_R),2);  //R����
			}
		}
	}
	variance_B=sqrt(variance_B/chromaticityShadowNum);
	variance_G=sqrt(variance_G/chromaticityShadowNum);
	variance_R=sqrt(variance_R/chromaticityShadowNum);

	//����RGB���������ĸߵ���ֵ
	thresholdH_B= m_B + 1.96*variance_B;  //B����
	thresholdL_B= m_B - 1.96*variance_B;
	thresholdH_G= m_G + 1.96*variance_G;  //G����
	thresholdL_G= m_G - 1.96*variance_G;
	thresholdH_R= m_R + 1.96*variance_R;  //R����
	thresholdL_R= m_R - 1.96*variance_R;

	cout<<endl;
	cout<<"--------------------BD������----------------------"<<endl;
	cout<<"BD_B��������"<<m_B<<endl;
	cout<<"BD_G��������"<<m_G<<endl;
	cout<<"BD_R��������"<<m_R<<endl;
	cout<<"BD_B�ķ��"<<variance_B<<endl;
	cout<<"BD_G�ķ��"<<variance_G<<endl;
	cout<<"BD_R�ķ��"<<variance_R<<endl;
	cout<<"B�ķ�����ֵ��"<<thresholdL_B<<"\t"<<thresholdH_B<<endl;
	cout<<"G�ķ�����ֵ��"<<thresholdL_G<<"\t"<<thresholdH_G<<endl;
	cout<<"R�ķ�����ֵ��"<<thresholdL_R<<"\t"<<thresholdH_R<<endl;

	brightnessMat=chromaticityMat.clone();   //�����brightnessMat������chromaticityMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	namedWindow("�Աȣ�ɫ�Ȳ�����",WINDOW_NORMAL);
	imshow("�Աȣ�ɫ�Ȳ�����", chromaticityMat);
	waitKey(0);
	//ɫ�Ȳ�����������Ϊ��ɫ�������ɫ����Ӱ��ɫ�����Ȳ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			//���Ȳ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//BD����ֵ�����ڣ�������Ӱ
				//if( (bd_B[i][j]>thresholdL_B && bd_B[i][j]<thresholdH_B) || (bd_G[i][j]>thresholdL_G && bd_G[i][j]<thresholdH_G) || (bd_R[i][j]>thresholdL_R && bd_R[i][j]<thresholdH_R) )
				if( (bd_B[i][j]>thresholdL_B && bd_B[i][j]<thresholdH_B) && (bd_G[i][j]>thresholdL_G && bd_G[i][j]<thresholdH_G) && (bd_R[i][j]>thresholdL_R && bd_R[i][j]<thresholdH_R) )
				{
					continue;
				}  
				else
				{
					brightnessMat.at<Vec3b>(i,j)[0]=255;   //�¼�������Ϊ��ɫ
					brightnessMat.at<Vec3b>(i,j)[1]=0;
					brightnessMat.at<Vec3b>(i,j)[2]=0;
				}
			}
		}
	}
	//�˲���ʾ��Ϊ�˽����Ȳ�������ɫ�Ȳ��������жԱ�
	namedWindow("�Աȣ����Ȳ�����",WINDOW_NORMAL);
	imshow("�Աȣ����Ȳ�����", brightnessMat);
	waitKey(0);
	destroyWindow("�Աȣ�ɫ�Ȳ�����");
	destroyWindow("�Աȣ����Ȳ�����");

	//����Ա�ͼƬ
	imwrite("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.jpg", brightnessMat);

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
	imwrite("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.jpg", brightnessMat);

	return 0;
}

//�ֲ����ȱ�
int localRelation()
{
	cout<<"-------------�ֲ�ǿ�ȱȼ����Ӱ------------------"<<endl;
	//��ʼ��Qֵ
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			q_B[i][j]=0;
			q_G[i][j]=0;
			q_R[i][j]=0;
		}
	}

	localMat=brightnessMat.clone();   //�����localMat������brightnessMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	namedWindow("�Աȣ�ɫ��+���Ȳ�����",WINDOW_NORMAL);
	imshow("�Աȣ�ɫ��+���Ȳ�����", brightnessMat);
	waitKey(0);

	//ɫ�Ȳ�+���Ȳ�����������Ϊ��ɫ�������ɫ����Ӱ��ɫ���ֲ����ȱ�ֻ������Ӱ��ѡ��������ĳ���������廹����Ӱ
	for(int i=1;i<brightnessMat.rows-1;i++)  //ע��ͼƬ��Ե�������Qֵ��ע��i��j��ȡֵ��Χ
	{
		for(int j=1;j<brightnessMat.cols-1;j++)
		{
			//�ֲ����ȱ�ֻ������Ӱ��ѡ������ɫ��������ĳ���������廹����Ӱ
			if( abs(brightnessMat.at<Vec3b>(i,j)[0]-0)==0 && abs(brightnessMat.at<Vec3b>(i,j)[1]-255)==0 && abs(brightnessMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//�ų���Ӱ��Ե: ���ص�����ҲҪ������Ӱ
				if ( (brightnessMat.at<Vec3b>(i,j-1)[0]==0 && brightnessMat.at<Vec3b>(i,j-1)[1]==255 && brightnessMat.at<Vec3b>(i,j-1)[2]==0) && (brightnessMat.at<Vec3b>(i+1,j)[0]==0 && brightnessMat.at<Vec3b>(i+1,j)[1]==255 && brightnessMat.at<Vec3b>(i+1,j)[2]==0) && (brightnessMat.at<Vec3b>(i,j+1)[0]==0 && brightnessMat.at<Vec3b>(i,j+1)[1]==255 && brightnessMat.at<Vec3b>(i,j+1)[2]==0) && (brightnessMat.at<Vec3b>(i-1,j)[0]==0 && brightnessMat.at<Vec3b>(i-1,j)[1]==255 && brightnessMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					q_B[i][j]= pow((bd_B[i][j-1]-m_B)/variance_B,2)+ pow((bd_B[i+1][j]-m_B)/variance_B,2)+ pow((bd_B[i][j+1]-m_B)/variance_B,2)+ pow((bd_B[i-1][j]-m_B)/variance_B,2);
					q_G[i][j]= pow((bd_G[i][j-1]-m_G)/variance_G,2)+ pow((bd_B[i+1][j]-m_G)/variance_G,2)+ pow((bd_B[i][j+1]-m_G)/variance_G,2)+ pow((bd_B[i-1][j]-m_G)/variance_G,2);
					q_R[i][j]= pow((bd_B[i][j-1]-m_R)/variance_R,2)+ pow((bd_B[i+1][j]-m_R)/variance_R,2)+ pow((bd_R[i][j+1]-m_R)/variance_R,2)+ pow((bd_B[i-1][j]-m_R)/variance_R,2);
				}
			}
		}
	}
	
	//����Qֵ����ɫ��+���ȼ�⵽����Ӱ���ٴ����ж�
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
						localMat.at<Vec3b>(i,j)[1]=0;
						localMat.at<Vec3b>(i,j)[2]=255;
					}
				}
			}
		}
	}

	//�˲���ʾ��Ϊ�˽��ֲ��Աȼ������ɫ��+���Ȳ��������жԱ�
	namedWindow("�Աȣ��ֲ��Աȼ����",WINDOW_NORMAL);
	imshow("�Աȣ��ֲ��Աȼ����", localMat);
	waitKey(0);
	destroyWindow("�Աȣ��ֲ��Աȼ����");
	destroyWindow("�Աȣ����Ȳ�����");

	//����Ա�ͼƬ
	imwrite("G:\\Code-Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.jpg", localMat);

	//����ɫ��+����+�ֲ��ԱȵĽ����ͳһ��ɫ����Ӱ��ɫ�������ɫ
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			if( abs(localMat.at<Vec3b>(i,j)[0]-255)==0 && abs(localMat.at<Vec3b>(i,j)[1]-0)==0 && abs(localMat.at<Vec3b>(i,j)[2]-255)==0 )
			{
				localMat.at<Vec3b>(i,j)[0]=0;   	//���ϴ��¼�⵽��������ɫ�ɷ�ɫ��Ϊ��ɫ����
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
	imwrite("G:\\Code-Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.jpg", localMat);

	return 0;
}

//���ڵ�����Ӱ������
int spatialAjustment()
{
	spacialMat=localMat.clone();   //�����spacialMat������localMat���γ�һ���µ�ͼ����������໥û��Ӱ��
	namedWindow("�Աȣ�ɫ��+���Ȳ�+�ֲ������",WINDOW_NORMAL);
	imshow("�Աȣ�ɫ��+���Ȳ�+�ֲ������", localMat);
	waitKey(0);



	return 0;
}

//��Ӱ����㷨
int shadowDetection()
{
	//step1. ɫ�Ȳ���Ӱ���
	chromaticityDiffer();

	//step2. ���Ȳ���Ӱ���
	brightnessDiffer();

	//step3. �ֲ����ȱ�
	localRelation();

	//step4. ���ڵ�����Ӱ������
	spatialAjustment();

	return 0;
}