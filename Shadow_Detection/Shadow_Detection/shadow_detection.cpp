/*
------------------------------------------------
Author: CIEL
Date: 2017/03/01
Function: 阴影检测
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
#define WIDTH 1408  //HoloLens图像宽度
#define HEIGHT 792  //HoloLens图像高度
*/

/*
#define WIDTH 320  //图像宽度
#define HEIGHT 240  //图像高度
*/

#define WIDTH 1216  //HoloLens视频截图宽度
#define HEIGHT 684  //HoloLens视频截图高度

Mat sceneMat, backgroundMat;
Mat chromaticityMat, brightnessMat, localMat, spacialMat;  //存储每步阴影检测结果

int sceneRGB_B[WIDTH][HEIGHT],sceneRGB_G[WIDTH][HEIGHT],sceneRGB_R[WIDTH][HEIGHT];  //前景图像的RGB分量
int backgroundRGB_B[WIDTH][HEIGHT],backgroundRGB_G[WIDTH][HEIGHT],backgroundRGB_R[WIDTH][HEIGHT];  //背景图像的RGB分量
double sceneNorm[WIDTH][HEIGHT],backgroundNorm[WIDTH][HEIGHT];  //前景图像和背景图像每个像素RGB分量的2范数
double cd_B[WIDTH][HEIGHT],cd_G[WIDTH][HEIGHT],cd_R[WIDTH][HEIGHT];  //色度差的RGB分量
double bd_B[WIDTH][HEIGHT],bd_G[WIDTH][HEIGHT],bd_R[WIDTH][HEIGHT];  //亮度差的RGB分量
double q_B[WIDTH][HEIGHT], q_G[WIDTH][HEIGHT], q_R[WIDTH][HEIGHT]; //RGB三个分量的Q值

double m_B, m_G, m_R;  //RGB三个分量的期望
double variance_B, variance_G, variance_R;   //RGB三个分量的方差
double thresholdH_B, thresholdL_B, thresholdH_G, thresholdL_G, thresholdH_R, thresholdL_R;  //RGB三个分量的高低阈值

int chromaticityShadowNum;  //色度差检测到的阴影像素数

//求向量的2范数
double norm2(int b,int g,int r)
{
	double norm=0;
	norm=sqrt(b*b+g*g+r*r);
	return norm;
}

//计算前景图像与背景图像的色度差
//假设输入的前景图片已经处理为背景是黄色，前景非黄色
int chromaticityDiffer()
{
	sceneMat=imread("G:\\Code-Shadow Detection\\Data\\Foreground\\20170228111043_fore.jpg");  //最开始提取到的前景图像
	backgroundMat=imread("G:\\Code-Shadow Detection\\Data\\Background\\20170228111043_back.jpg");  //背景图像

	namedWindow("前景图");
	imshow("前景图", sceneMat);
	waitKey(0);

	//遍历前景图像的每个像素，注：RGB三个分量都要计算
	for(int i=0;i<sceneMat.rows;i++)
	{
		const Vec3b* scenePoint=sceneMat.ptr <Vec3b>(i);  //Vec3b是一个三元向量的数据结构，正好能够表示RGB的三个分量
		for(int j=0;j<sceneMat.cols;j++)
		{
			Vec3b intensity=*(scenePoint+j);
			sceneRGB_B[i][j]=intensity[0];
			sceneRGB_G[i][j]=intensity[1];
			sceneRGB_R[i][j]=intensity[2];
		}
	}
	//遍历背景图像的每个像素，注：RGB三个分量都要计算
	for(int i=0;i<backgroundMat.rows;i++)
	{
		const Vec3b* backgrounPoint=backgroundMat.ptr <Vec3b>(i);  //Vec3b是一个三元向量的数据结构，正好能够表示RGB的三个分量
		for(int j=0;j<backgroundMat.cols;j++)
		{
			Vec3b intensity=*(backgrounPoint+j);
			backgroundRGB_B[i][j]=intensity[0];
			backgroundRGB_G[i][j]=intensity[1];
			backgroundRGB_R[i][j]=intensity[2];
		}
	}

	//计算前景图像和背景图像每个像素RGB分量的2范数
	for(int i=0;i<backgroundMat.rows;i++)
	{
		for(int j=0;j<backgroundMat.cols;j++)
		{
			sceneNorm[i][j]=norm2(sceneRGB_B[i][j],sceneRGB_G[i][j],sceneRGB_R[i][j]);
			backgroundNorm[i][j]=norm2(backgroundRGB_B[i][j],backgroundRGB_G[i][j],backgroundRGB_R[i][j]);
		}
	}

	//计算前景图像与背景图像每个像素的色度差，注：RGB三个分量都要计算
	//注：可能会出现分母为零的情况！！！！一定要判断！！！！如果分母为零，则将其设为无穷小！！！！
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

	//将每个像素的CD值保存到txt文件中
	//B分量
	ofstream out_cdB("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_B.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdB<<cd_B[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_cdB<<endl;   //每行输出结束，添加换行
	}
	//G分量
	ofstream out_cdG("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_G.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdG<<cd_G[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_cdG<<endl;   //每行输出结束，添加换行
	}
	//R分量
	ofstream out_cdR("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_R.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdR<<cd_R[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_cdR<<endl;   //每行输出结束，添加换行
	}
	out_cdB.close();
	out_cdG.close();
	out_cdR.close();


	//计算CD的期望
	int cdNum_B=0, cdNum_G=0, cdNum_R=0; //RGB三个分量符合区间为[-0.2,0.2]的CD的个数
	m_B=0;
	m_G=0;
	m_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_B++;
				m_B=m_B+cd_B[i][j];
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_G++;
				m_G=m_G+cd_G[i][j];
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_R++;
				m_R=m_R+cd_R[i][j];
			}
		}
	}
	m_B=m_B/cdNum_B;
	m_G=m_G/cdNum_G;
	m_R=m_R/cdNum_R;

	//计算CD的方差
	variance_B=0;
	variance_G=0;
	variance_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B分量：保留符合区间为[-0.2,0.2]的CD
			{
				variance_B=variance_B + pow((cd_B[i][j]-m_B),2);
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G分量：保留符合区间为[-0.2,0.2]的CD
			{
				variance_G=variance_G + pow((cd_G[i][j]-m_G),2);
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R分量：保留符合区间为[-0.2,0.2]的CD
			{
				variance_R=variance_R + pow((cd_R[i][j]-m_R),2);
			}
		}
	}
	variance_B=sqrt(variance_B/cdNum_B);
	variance_G=sqrt(variance_G/cdNum_G);
	variance_R=sqrt(variance_R/cdNum_R);

	//计算RGB三个分量的高低阈值
	thresholdH_B= m_B + 1.96*variance_B;  //B分量
	thresholdL_B= m_B - 1.96*variance_B;
	thresholdH_G= m_G + 1.96*variance_G;  //G分量
	thresholdL_G= m_G - 1.96*variance_G;
	thresholdH_R= m_R + 1.96*variance_R;  //R分量
	thresholdL_R= m_R - 1.96*variance_R;

	cout<<"--------------------CD计算结果----------------------"<<endl;
	cout<<"[-0.2,0.2]cdNum_B："<<cdNum_B<<endl;
	cout<<"[-0.2,0.2]cdNum_G："<<cdNum_G<<endl;
	cout<<"[-0.2,0.2]cdNum_R："<<cdNum_R<<endl;
	cout<<"CD_B的期望："<<m_B<<endl;
	cout<<"CD_G的期望："<<m_G<<endl;
	cout<<"CD_R的期望："<<m_R<<endl;
	cout<<"CD_B的方差："<<variance_B<<endl;
	cout<<"CD_G的方差："<<variance_G<<endl;
	cout<<"CD_R的方差："<<variance_R<<endl;
	cout<<"B的分类阈值："<<thresholdL_B<<"\t"<<thresholdH_B<<endl;
	cout<<"G的分类阈值："<<thresholdL_G<<"\t"<<thresholdH_G<<endl;
	cout<<"R的分类阈值："<<thresholdL_R<<"\t"<<thresholdH_R<<endl;

	chromaticityMat=sceneMat.clone();   //深拷贝：chromaticityMat拷贝了sceneMat，形成一个新的图像矩阵，两者相互没有影响	
	chromaticityShadowNum=0;  //色度差检测到的阴影像素数
	//计算BD的期望
	m_B=0;
	m_G=0;
	m_R=0;
	//初始化BD
	for(int i=0;i<chromaticityMat.rows;i++)
	{
		for(int j=0;j<chromaticityMat.cols;j++)
		{
			bd_B[i][j]=0;
			bd_G[i][j]=0;
			bd_R[i][j]=0;
		}
	}
	//前景图像背景为黄色，物体候选区非黄，所以只需在物体候选区中区分某像素是物体还是阴影
	for(int i=0;i<chromaticityMat.rows;i++)
	{
		for(int j=0;j<chromaticityMat.cols;j++)
		{		
			//黄色是背景:允许像素有误差
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)<=30 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)<=30 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-255)<=30 )
				continue;  
			else
			{
				if(cd_B[i][j]>0.2 || cd_G[i][j]>0.2  ||cd_R[i][j]>0.2)  //CD>0.2,必为物体
				{
					chromaticityMat.at<Vec3b>(i,j)[0]=0;   //物体为红色
					chromaticityMat.at<Vec3b>(i,j)[1]=0;
					chromaticityMat.at<Vec3b>(i,j)[2]=255;
				}
				else
				{
					//CD在阈值区间内，属于阴影
					if( (cd_B[i][j]>thresholdL_B && cd_B[i][j]<thresholdH_B) || (cd_G[i][j]>thresholdL_G && cd_G[i][j]<thresholdH_G) || (cd_R[i][j]>thresholdL_R && cd_R[i][j]<thresholdH_R) )
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;  //阴影为绿色
						chromaticityMat.at<Vec3b>(i,j)[1]=255;
						chromaticityMat.at<Vec3b>(i,j)[2]=0;

						//统计色度差检测到的阴影像素个数,计算它们的BD值
						chromaticityShadowNum++;  
						//注：可能会出现分母为零的情况！！！！一定要判断！！！！如果分母为零，则将其设为无穷小！！！！
						if(backgroundRGB_B[i][j]==0)
							backgroundRGB_B[i][j]=INT_MIN;
						if(backgroundRGB_G[i][j]==0)
							backgroundRGB_G[i][j]=INT_MIN;
						if(backgroundRGB_R[i][j]==0)
							backgroundRGB_R[i][j]=INT_MIN;
						bd_B[i][j]=sceneRGB_B[i][j]/backgroundRGB_B[i][j];
						bd_G[i][j]=sceneRGB_G[i][j]/backgroundRGB_G[i][j];
						bd_R[i][j]=sceneRGB_R[i][j]/backgroundRGB_R[i][j];
						m_B=m_B+bd_B[i][j];  //B分量
						m_G=m_G+bd_G[i][j];  //G分量
						m_R=m_R+bd_R[i][j];  //R分量
					}  
					else
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;   //物体为红色
						chromaticityMat.at<Vec3b>(i,j)[1]=0;
						chromaticityMat.at<Vec3b>(i,j)[2]=255;
					}

				}
			}

		}
	}
	cout<<"色度差检测到的阴影像素数："<<chromaticityShadowNum<<endl;

	namedWindow("色度差检测结果",WINDOW_NORMAL);
	imshow("色度差检测结果", chromaticityMat);
	waitKey(0);
	destroyWindow("前景图");
	destroyWindow("色度差检测结果");

	//保存图片
	//	imwrite("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\example_chromaticity.jpg", result);
	imwrite("G:\\Code-Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\20170228111043_chromaticity.jpg", chromaticityMat);

	return 0;
}


//计算前景图像与背景图像的亮度差
//输入的前景图片已经处理为背景是黄色，阴影绿色，物体红色
int brightnessDiffer()
{
	//按照论文，只对色度差检测到的阴影继续筛选
	//将每个像素的BD值保存到txt文件中
	//B分量
	ofstream out_bdB("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_B.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdB<<bd_B[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_bdB<<endl;   //每行输出结束，添加换行
	}
	//G分量
	ofstream out_bdG("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_G.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdG<<bd_G[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_bdG<<endl;   //每行输出结束，添加换行
	}
	//R分量
	ofstream out_bdR("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_R.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdR<<bd_R[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_bdR<<endl;   //每行输出结束，添加换行
	}
	out_bdB.close();
	out_bdG.close();
	out_bdR.close();

	//计算BD的期望
	m_B=m_B/chromaticityShadowNum;
	m_G=m_G/chromaticityShadowNum;
	m_R=m_R/chromaticityShadowNum;

	//计算BD的方差
	variance_B=0;
	variance_G=0;
	variance_R=0;
	for(int i=0;i<sceneMat.rows; i++)
	{
		for(int j=0; j<sceneMat.cols; j++)
		{
			if(chromaticityMat.at<Vec3b>(i,j)[0]==0 && chromaticityMat.at<Vec3b>(i,j)[1]==255 && chromaticityMat.at<Vec3b>(i,j)[0]==0)
			{
				variance_B=variance_B + pow((bd_B[i][j]-m_B),2);  //B分量
				variance_G=variance_G + pow((bd_G[i][j]-m_G),2);  //G分量
				variance_R=variance_R + pow((bd_R[i][j]-m_R),2);  //R分量
			}
		}
	}
	variance_B=sqrt(variance_B/chromaticityShadowNum);
	variance_G=sqrt(variance_G/chromaticityShadowNum);
	variance_R=sqrt(variance_R/chromaticityShadowNum);

	//计算RGB三个分量的高低阈值
	thresholdH_B= m_B + 1.96*variance_B;  //B分量
	thresholdL_B= m_B - 1.96*variance_B;
	thresholdH_G= m_G + 1.96*variance_G;  //G分量
	thresholdL_G= m_G - 1.96*variance_G;
	thresholdH_R= m_R + 1.96*variance_R;  //R分量
	thresholdL_R= m_R - 1.96*variance_R;

	cout<<endl;
	cout<<"--------------------BD计算结果----------------------"<<endl;
	cout<<"BD_B的期望："<<m_B<<endl;
	cout<<"BD_G的期望："<<m_G<<endl;
	cout<<"BD_R的期望："<<m_R<<endl;
	cout<<"BD_B的方差："<<variance_B<<endl;
	cout<<"BD_G的方差："<<variance_G<<endl;
	cout<<"BD_R的方差："<<variance_R<<endl;
	cout<<"B的分类阈值："<<thresholdL_B<<"\t"<<thresholdH_B<<endl;
	cout<<"G的分类阈值："<<thresholdL_G<<"\t"<<thresholdH_G<<endl;
	cout<<"R的分类阈值："<<thresholdL_R<<"\t"<<thresholdH_R<<endl;

	brightnessMat=chromaticityMat.clone();   //深拷贝：brightnessMat拷贝了chromaticityMat，形成一个新的图像矩阵，两者相互没有影响
	namedWindow("对比：色度差检测结果",WINDOW_NORMAL);
	imshow("对比：色度差检测结果", chromaticityMat);
	waitKey(0);
	//色度差检测结果：背景为黄色，物体红色，阴影绿色。亮度差只需在阴影候选区中区分某像素是物体还是阴影
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			//亮度差只需在阴影候选区中区分某像素是物体还是阴影
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//BD在阈值区间内，属于阴影
				//if( (bd_B[i][j]>thresholdL_B && bd_B[i][j]<thresholdH_B) || (bd_G[i][j]>thresholdL_G && bd_G[i][j]<thresholdH_G) || (bd_R[i][j]>thresholdL_R && bd_R[i][j]<thresholdH_R) )
				if( (bd_B[i][j]>thresholdL_B && bd_B[i][j]<thresholdH_B) && (bd_G[i][j]>thresholdL_G && bd_G[i][j]<thresholdH_G) && (bd_R[i][j]>thresholdL_R && bd_R[i][j]<thresholdH_R) )
				{
					continue;
				}  
				else
				{
					brightnessMat.at<Vec3b>(i,j)[0]=255;   //新检测的物体为蓝色
					brightnessMat.at<Vec3b>(i,j)[1]=0;
					brightnessMat.at<Vec3b>(i,j)[2]=0;
				}
			}
		}
	}
	//此步显示是为了将亮度差检测结果与色度差检测结果进行对比
	namedWindow("对比：亮度差检测结果",WINDOW_NORMAL);
	imshow("对比：亮度差检测结果", brightnessMat);
	waitKey(0);
	destroyWindow("对比：色度差检测结果");
	destroyWindow("对比：亮度差检测结果");

	//保存对比图片
	imwrite("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.jpg", brightnessMat);

	//保存色度+亮度的结果，统一颜色：阴影绿色，物体红色
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			if( abs(brightnessMat.at<Vec3b>(i,j)[0]-255)==0 && abs(brightnessMat.at<Vec3b>(i,j)[1]-0)==0 && abs(brightnessMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				brightnessMat.at<Vec3b>(i,j)[0]=0;   	//将上次新检测到的物体颜色由蓝色改为红色即可
				brightnessMat.at<Vec3b>(i,j)[1]=0;
				brightnessMat.at<Vec3b>(i,j)[2]=255;
			}
		}
	}
	namedWindow("色度+亮度差检测结果",WINDOW_NORMAL);
	imshow("色度+亮度差检测结果", brightnessMat);
	waitKey(0);
	destroyWindow("色度+亮度差检测结果");
	//保存进一步检测的图片
	imwrite("G:\\Code-Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.jpg", brightnessMat);

	return 0;
}

//局部亮度比
int localRelation()
{
	cout<<"-------------局部强度比检测阴影------------------"<<endl;
	//初始化Q值
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			q_B[i][j]=0;
			q_G[i][j]=0;
			q_R[i][j]=0;
		}
	}

	localMat=brightnessMat.clone();   //深拷贝：localMat拷贝了brightnessMat，形成一个新的图像矩阵，两者相互没有影响
	namedWindow("对比：色度+亮度差检测结果",WINDOW_NORMAL);
	imshow("对比：色度+亮度差检测结果", brightnessMat);
	waitKey(0);

	//色度差+亮度差检测结果：背景为黄色，物体红色，阴影绿色。局部亮度比只需在阴影候选区中区分某像素是物体还是阴影
	for(int i=1;i<brightnessMat.rows-1;i++)  //注：图片边缘无需计算Q值，注意i和j的取值范围
	{
		for(int j=1;j<brightnessMat.cols-1;j++)
		{
			//局部亮度比只需在阴影候选区（绿色）中区分某像素是物体还是阴影
			if( abs(brightnessMat.at<Vec3b>(i,j)[0]-0)==0 && abs(brightnessMat.at<Vec3b>(i,j)[1]-255)==0 && abs(brightnessMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//排除阴影边缘: 像素的邻域也要属于阴影
				if ( (brightnessMat.at<Vec3b>(i,j-1)[0]==0 && brightnessMat.at<Vec3b>(i,j-1)[1]==255 && brightnessMat.at<Vec3b>(i,j-1)[2]==0) && (brightnessMat.at<Vec3b>(i+1,j)[0]==0 && brightnessMat.at<Vec3b>(i+1,j)[1]==255 && brightnessMat.at<Vec3b>(i+1,j)[2]==0) && (brightnessMat.at<Vec3b>(i,j+1)[0]==0 && brightnessMat.at<Vec3b>(i,j+1)[1]==255 && brightnessMat.at<Vec3b>(i,j+1)[2]==0) && (brightnessMat.at<Vec3b>(i-1,j)[0]==0 && brightnessMat.at<Vec3b>(i-1,j)[1]==255 && brightnessMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					q_B[i][j]= pow((bd_B[i][j-1]-m_B)/variance_B,2)+ pow((bd_B[i+1][j]-m_B)/variance_B,2)+ pow((bd_B[i][j+1]-m_B)/variance_B,2)+ pow((bd_B[i-1][j]-m_B)/variance_B,2);
					q_G[i][j]= pow((bd_G[i][j-1]-m_G)/variance_G,2)+ pow((bd_B[i+1][j]-m_G)/variance_G,2)+ pow((bd_B[i][j+1]-m_G)/variance_G,2)+ pow((bd_B[i-1][j]-m_G)/variance_G,2);
					q_R[i][j]= pow((bd_B[i][j-1]-m_R)/variance_R,2)+ pow((bd_B[i+1][j]-m_R)/variance_R,2)+ pow((bd_R[i][j+1]-m_R)/variance_R,2)+ pow((bd_B[i-1][j]-m_R)/variance_R,2);
				}
			}
		}
	}
	
	//利用Q值，将色度+亮度检测到的阴影，再次做判断
	for(int i=1;i<localMat.rows-1;i++)  //注：图片边缘无需计算Q值，注意i和j的取值范围
	{
		for(int j=1;j<localMat.cols-1;j++)
		{
			//局部亮度比只需在阴影候选区（绿色）中区分某像素是物体还是阴影
			if( abs(localMat.at<Vec3b>(i,j)[0]-0)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//排除阴影边缘: 像素的邻域也要属于阴影
				if ( (localMat.at<Vec3b>(i,j-1)[0]==0 && localMat.at<Vec3b>(i,j-1)[1]==255 && localMat.at<Vec3b>(i,j-1)[2]==0) && (localMat.at<Vec3b>(i+1,j)[0]==0 && localMat.at<Vec3b>(i+1,j)[1]==255 && localMat.at<Vec3b>(i+1,j)[2]==0) && (localMat.at<Vec3b>(i,j+1)[0]==0 && localMat.at<Vec3b>(i,j+1)[1]==255 && localMat.at<Vec3b>(i,j+1)[2]==0) && (localMat.at<Vec3b>(i-1,j)[0]==0 && localMat.at<Vec3b>(i-1,j)[1]==255 && localMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					//if(q_B[i][j]<9.49 || q_G[i][j]<9.49 || q_R[i][j]<9.49) //阴影 
					if(q_B[i][j]<9.49 && q_G[i][j]<9.49 && q_R[i][j]<9.49) //阴影 
						continue;   
					else
					{
						localMat.at<Vec3b>(i,j)[0]=255;   //新检测的物体为粉色
						localMat.at<Vec3b>(i,j)[1]=0;
						localMat.at<Vec3b>(i,j)[2]=255;
					}
				}
			}
		}
	}

	//此步显示是为了将局部对比检测结果与色度+亮度差检测结果进行对比
	namedWindow("对比：局部对比检测结果",WINDOW_NORMAL);
	imshow("对比：局部对比检测结果", localMat);
	waitKey(0);
	destroyWindow("对比：局部对比检测结果");
	destroyWindow("对比：亮度差检测结果");

	//保存对比图片
	imwrite("G:\\Code-Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.jpg", localMat);

	//保存色度+亮度+局部对比的结果，统一颜色：阴影绿色，物体红色
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			if( abs(localMat.at<Vec3b>(i,j)[0]-255)==0 && abs(localMat.at<Vec3b>(i,j)[1]-0)==0 && abs(localMat.at<Vec3b>(i,j)[2]-255)==0 )
			{
				localMat.at<Vec3b>(i,j)[0]=0;   	//将上次新检测到的物体颜色由粉色改为红色即可
				localMat.at<Vec3b>(i,j)[1]=0;
				localMat.at<Vec3b>(i,j)[2]=255;
			}
		}
	}
	namedWindow("色度+亮度差+局部对比检测结果",WINDOW_NORMAL);
	imshow("色度+亮度差+局部对比检检测结果", localMat);
	waitKey(0);
	destroyWindow("色度+亮度差+局部对比检检测结果");
	//保存进一步检测的图片
	imwrite("G:\\Code-Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.jpg", localMat);

	return 0;
}

//基于调整阴影和物体
int spatialAjustment()
{
	spacialMat=localMat.clone();   //深拷贝：spacialMat拷贝了localMat，形成一个新的图像矩阵，两者相互没有影响
	namedWindow("对比：色度+亮度差+局部检测结果",WINDOW_NORMAL);
	imshow("对比：色度+亮度差+局部检测结果", localMat);
	waitKey(0);



	return 0;
}

//阴影检测算法
int shadowDetection()
{
	//step1. 色度差阴影检测
	chromaticityDiffer();

	//step2. 亮度差阴影检测
	brightnessDiffer();

	//step3. 局部亮度比
	localRelation();

	//step4. 基于调整阴影和物体
	spatialAjustment();

	return 0;
}