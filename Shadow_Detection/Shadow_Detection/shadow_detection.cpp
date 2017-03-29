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
#include <cv.h>
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

#define WINDOW_NAME1 "原始图窗口"
#define WINDOW_NAME2 "效果图窗口"

Mat sceneMat, backgroundMat;
Mat chromaticityMat, brightnessMat, localMat, spacialMat, spacialGrayMat;  //存储每步阴影检测结果

/*//宽高写反了
int sceneRGB_B[WIDTH][HEIGHT],sceneRGB_G[WIDTH][HEIGHT],sceneRGB_R[WIDTH][HEIGHT];  //前景图像的RGB分量
int backgroundRGB_B[WIDTH][HEIGHT],backgroundRGB_G[WIDTH][HEIGHT],backgroundRGB_R[WIDTH][HEIGHT];  //背景图像的RGB分量
double sceneNorm[WIDTH][HEIGHT],backgroundNorm[WIDTH][HEIGHT];  //前景图像和背景图像每个像素RGB分量的2范数
double cd_B[WIDTH][HEIGHT],cd_G[WIDTH][HEIGHT],cd_R[WIDTH][HEIGHT];  //色度差的RGB分量
double bd_B[WIDTH][HEIGHT],bd_G[WIDTH][HEIGHT],bd_R[WIDTH][HEIGHT];  //亮度差的RGB分量
double q_B[WIDTH][HEIGHT], q_G[WIDTH][HEIGHT], q_R[WIDTH][HEIGHT]; //RGB三个分量的Q值
*/
int sceneRGB_B[HEIGHT][WIDTH],sceneRGB_G[HEIGHT][WIDTH],sceneRGB_R[HEIGHT][WIDTH];  //前景图像的RGB分量
int backgroundRGB_B[HEIGHT][WIDTH],backgroundRGB_G[HEIGHT][WIDTH],backgroundRGB_R[HEIGHT][WIDTH];  //背景图像的RGB分量
double sceneNorm[HEIGHT][WIDTH],backgroundNorm[HEIGHT][WIDTH];  //前景图像和背景图像每个像素RGB分量的2范数
double cd_B[HEIGHT][WIDTH],cd_G[HEIGHT][WIDTH],cd_R[HEIGHT][WIDTH];  //色度差的RGB分量
double bd_B[HEIGHT][WIDTH],bd_G[HEIGHT][WIDTH],bd_R[HEIGHT][WIDTH];  //亮度差的RGB分量
double q_B[HEIGHT][WIDTH], q_G[HEIGHT][WIDTH], q_R[HEIGHT][WIDTH]; //RGB三个分量的Q值

//色度CD
double cd_m_B, cd_m_G, cd_m_R;  //RGB三个分量的期望
double cd_variance_B, cd_variance_G, cd_variance_R;   //RGB三个分量的方差
double cd_thresholdH_B, cd_thresholdL_B, cd_thresholdH_G, cd_thresholdL_G, cd_thresholdH_R, cd_thresholdL_R;  //RGB三个分量的高低阈值
//亮度BD
double bd_m_B, bd_m_G, bd_m_R;  //RGB三个分量的期望
double bd_variance_B, bd_variance_G, bd_variance_R;   //RGB三个分量的方差
double bd_thresholdH_B, bd_thresholdL_B, bd_thresholdH_G, bd_thresholdL_G, bd_thresholdH_R, bd_thresholdL_R;  //RGB三个分量的高低阈值

struct pixelInformation{   //结构体存放每个像素点的信息
	int category;  //判断像素点的种类。0，背景；1，物体；2，阴影
	int initColor_B;  //原图的RBG颜色-B
	int initColor_G;  //原图的RBG颜色-G
	int initColor_R;  //原图的RBG颜色-R
	int revise;  //判断该像素是否被修改。0，没有被修改；1，被修改
};
//struct pixelInformation graph[WIDTH][HEIGHT];
struct pixelInformation graph[HEIGHT][WIDTH];


int chromaticityShadowNum;  //色度差检测到的阴影像素数

int g_nThresh=100;
int g_maxThresh=255;
RNG g_rng(12345);
//vector<vector<Point>> g_vContours;
//vector<Vec4i> g_viHierarchy;
void on_ThreshChange(int, void*);

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
	sceneMat=imread("F:\\Code\\Shadow Detection\\Data\\Foreground\\20170228111043_fore.jpg");  //最开始提取到的前景图像
	backgroundMat=imread("F:\\Code\\Shadow Detection\\Data\\Background\\20170228111043_back.jpg");  //背景图像

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

			//初始化结构体像素类别，初始化认为每个像素都是背景
			graph[i][j].category=0;
			//初始化结构体颜色，颜色同前景图像
			graph[i][j].initColor_B=sceneRGB_B[i][j];
			graph[i][j].initColor_G=sceneRGB_G[i][j];
			graph[i][j].initColor_R=sceneRGB_R[i][j];
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
	ofstream out_cdB("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_B.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdB<<cd_B[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_cdB<<endl;   //每行输出结束，添加换行
	}
	//G分量
	ofstream out_cdG("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_G.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_cdG<<cd_G[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_cdG<<endl;   //每行输出结束，添加换行
	}
	//R分量
	ofstream out_cdR("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Statistics\\cd_R.txt");  //打开文件
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
	cd_m_B=0;
	cd_m_G=0;
	cd_m_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_B++;
				cd_m_B=cd_m_B+cd_B[i][j];
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_G++;
				cd_m_G=cd_m_G+cd_G[i][j];
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R分量：保留符合区间为[-0.2,0.2]的CD
			{
				cdNum_R++;
				cd_m_R=cd_m_R+cd_R[i][j];
			}
		}
	}
	cd_m_B=cd_m_B/cdNum_B;
	cd_m_G=cd_m_G/cdNum_G;
	cd_m_R=cd_m_R/cdNum_R;

	//计算CD的方差
	cd_variance_B=0;
	cd_variance_G=0;
	cd_variance_R=0;
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			if(cd_B[i][j]>=-0.2 && cd_B[i][j]<=0.2)   //B分量：保留符合区间为[-0.2,0.2]的CD
			{
				cd_variance_B=cd_variance_B + pow((cd_B[i][j]-cd_m_B),2);
			}
			if(cd_G[i][j]>=-0.2 && cd_G[i][j]<=0.2)   //G分量：保留符合区间为[-0.2,0.2]的CD
			{
				cd_variance_G=cd_variance_G + pow((cd_G[i][j]-cd_m_G),2);
			}
			if(cd_R[i][j]>=-0.2 && cd_R[i][j]<=0.2)   //R分量：保留符合区间为[-0.2,0.2]的CD
			{
				cd_variance_R=cd_variance_R + pow((cd_R[i][j]-cd_m_R),2);
			}
		}
	}
	cd_variance_B=sqrt(cd_variance_B/cdNum_B);
	cd_variance_G=sqrt(cd_variance_G/cdNum_G);
	cd_variance_R=sqrt(cd_variance_R/cdNum_R);

	//计算RGB三个分量的高低阈值
	cd_thresholdH_B= cd_m_B + 1.96*cd_variance_B;  //B分量
	cd_thresholdL_B= cd_m_B - 1.96*cd_variance_B;
	cd_thresholdH_G= cd_m_G + 1.96*cd_variance_G;  //G分量
	cd_thresholdL_G= cd_m_G - 1.96*cd_variance_G;
	cd_thresholdH_R= cd_m_R + 1.96*cd_variance_R;  //R分量
	cd_thresholdL_R= cd_m_R - 1.96*cd_variance_R;

	cout<<"--------------------CD计算结果----------------------"<<endl;
	cout<<"[-0.2,0.2]cdNum_B："<<cdNum_B<<endl;
	cout<<"[-0.2,0.2]cdNum_G："<<cdNum_G<<endl;
	cout<<"[-0.2,0.2]cdNum_R："<<cdNum_R<<endl;
	cout<<"CD_B的期望："<<cd_m_B<<endl;
	cout<<"CD_G的期望："<<cd_m_G<<endl;
	cout<<"CD_R的期望："<<cd_m_R<<endl;
	cout<<"CD_B的方差："<<cd_variance_B<<endl;
	cout<<"CD_G的方差："<<cd_variance_G<<endl;
	cout<<"CD_R的方差："<<cd_variance_R<<endl;
	cout<<"B的分类阈值："<<cd_thresholdL_B<<"\t"<<cd_thresholdH_B<<endl;
	cout<<"G的分类阈值："<<cd_thresholdL_G<<"\t"<<cd_thresholdH_G<<endl;
	cout<<"R的分类阈值："<<cd_thresholdL_R<<"\t"<<cd_thresholdH_R<<endl;

	chromaticityMat=sceneMat.clone();   //深拷贝：chromaticityMat拷贝了sceneMat，形成一个新的图像矩阵，两者相互没有影响	
	chromaticityShadowNum=0;  //色度差检测到的阴影像素数
	//计算BD的期望
	bd_m_B=0;
	bd_m_G=0;
	bd_m_R=0;
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

					//修改结构体中相应的信息
					graph[i][j].category=1;
					graph[i][j].initColor_B=0;
					graph[i][j].initColor_G=0;
					graph[i][j].initColor_R=255;
				}
				else
				{
					//CD在阈值区间内，属于阴影
					if( (cd_B[i][j]>cd_thresholdL_B && cd_B[i][j]<cd_thresholdH_B) || (cd_G[i][j]>cd_thresholdL_G && cd_G[i][j]<cd_thresholdH_G) || (cd_R[i][j]>cd_thresholdL_R && cd_R[i][j]<cd_thresholdH_R) )
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;  //阴影为绿色
						chromaticityMat.at<Vec3b>(i,j)[1]=255;
						chromaticityMat.at<Vec3b>(i,j)[2]=0;

						//修改结构体中相应的信息
						graph[i][j].category=2;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=255;
						graph[i][j].initColor_R=0;

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
						bd_m_B=bd_m_B+bd_B[i][j];  //B分量
						bd_m_G=bd_m_G+bd_G[i][j];  //G分量
						bd_m_R=bd_m_R+bd_R[i][j];  //R分量
					}  
					else
					{
						chromaticityMat.at<Vec3b>(i,j)[0]=0;   //物体为红色
						chromaticityMat.at<Vec3b>(i,j)[1]=0;
						chromaticityMat.at<Vec3b>(i,j)[2]=255;

						//修改结构体中相应的信息
						graph[i][j].category=1;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=0;
						graph[i][j].initColor_R=255;
					}

				}
			}

		}
	}
	cout<<"色度差检测到的阴影像素数："<<chromaticityShadowNum<<endl;

	/*	//测试struct存储的阴影信息是否与色度差检测结果一致
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
	for(int j=0;j<WIDTH;j++)
	{	
	if(graph[i][j].category==2)
	testNum++;
	}
	}
	cout<<"结构体中存储的阴影数为:"<<testNum<<endl;

	cout<<"row="<<sceneMat.rows<<endl;
	cout<<"col="<<sceneMat.cols<<endl;
	*/

	namedWindow("色度差检测结果",WINDOW_NORMAL);
	imshow("色度差检测结果", chromaticityMat);
	waitKey(0);
	destroyWindow("前景图");
	destroyWindow("色度差检测结果");

	//保存图片
	//保存为bmp格式，图片不会有压缩；保存为jpg格式，图片会有压缩，即使把CV_IMWRITE_JPEG_QUALITY调整为100也不行
	vector<int>imwriteJPGquality;
	imwriteJPGquality.push_back(CV_IMWRITE_JPEG_QUALITY);   //JPG格式图片的质量
	imwriteJPGquality.push_back(100);
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\201702281110043_chromaticity.jpg", chromaticityMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Chromaticity Difference\\Chromaticity Differ Result\\201702281110043_chromaticity.bmp", chromaticityMat);
	return 0;
}


//计算前景图像与背景图像的亮度差
//输入的前景图片已经处理为背景是黄色，阴影绿色，物体红色
int brightnessDiffer()
{
	//按照论文，只对色度差检测到的阴影继续筛选
	//将每个像素的BD值保存到txt文件中
	//B分量
	ofstream out_bdB("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_B.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdB<<bd_B[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_bdB<<endl;   //每行输出结束，添加换行
	}
	//G分量
	ofstream out_bdG("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_G.txt");  //打开文件
	for(int i=0;i<sceneMat.rows;i++)
	{
		for(int j=0;j<sceneMat.cols;j++)
		{
			out_bdG<<bd_G[i][j]<<"\t";  //将每个元素写入文件，以Tab分隔   注：如果结果出现了-1.#IND则表示很小，不确定
		}
		out_bdG<<endl;   //每行输出结束，添加换行
	}
	//R分量
	ofstream out_bdR("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Statistics\\bd_R.txt");  //打开文件
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
	bd_m_B=bd_m_B/chromaticityShadowNum;
	bd_m_G=bd_m_G/chromaticityShadowNum;
	bd_m_R=bd_m_R/chromaticityShadowNum;

	//计算BD的方差
	bd_variance_B=0;
	bd_variance_G=0;
	bd_variance_R=0;
	for(int i=0;i<sceneMat.rows; i++)
	{
		for(int j=0; j<sceneMat.cols; j++)
		{
			if(chromaticityMat.at<Vec3b>(i,j)[0]==0 && chromaticityMat.at<Vec3b>(i,j)[1]==255 && chromaticityMat.at<Vec3b>(i,j)[0]==0)
			{
				bd_variance_B=bd_variance_B + pow((bd_B[i][j]-bd_m_B),2);  //B分量
				bd_variance_G=bd_variance_G + pow((bd_G[i][j]-bd_m_G),2);  //G分量
				bd_variance_R=bd_variance_R + pow((bd_R[i][j]-bd_m_R),2);  //R分量
			}
		}
	}
	bd_variance_B=sqrt(bd_variance_B/chromaticityShadowNum);
	bd_variance_G=sqrt(bd_variance_G/chromaticityShadowNum);
	bd_variance_R=sqrt(bd_variance_R/chromaticityShadowNum);

	//计算RGB三个分量的高低阈值
	bd_thresholdH_B= bd_m_B + 1.96*bd_variance_B;  //B分量
	bd_thresholdL_B= bd_m_B - 1.96*bd_variance_B;
	bd_thresholdH_G= bd_m_G + 1.96*bd_variance_G;  //G分量
	bd_thresholdL_G= bd_m_G - 1.96*bd_variance_G;
	bd_thresholdH_R= bd_m_R + 1.96*bd_variance_R;  //R分量
	bd_thresholdL_R= bd_m_R - 1.96*bd_variance_R;

	cout<<endl;
	cout<<"--------------------BD计算结果----------------------"<<endl;
	cout<<"BD_B的期望："<<bd_m_B<<endl;
	cout<<"BD_G的期望："<<bd_m_G<<endl;
	cout<<"BD_R的期望："<<bd_m_R<<endl;
	cout<<"BD_B的方差："<<bd_variance_B<<endl;
	cout<<"BD_G的方差："<<bd_variance_G<<endl;
	cout<<"BD_R的方差："<<bd_variance_R<<endl;
	cout<<"B的分类阈值："<<bd_thresholdL_B<<"\t"<<bd_thresholdH_B<<endl;
	cout<<"G的分类阈值："<<bd_thresholdL_G<<"\t"<<bd_thresholdH_G<<endl;
	cout<<"R的分类阈值："<<bd_thresholdL_R<<"\t"<<bd_thresholdH_R<<endl;

	brightnessMat=chromaticityMat.clone();   //深拷贝：brightnessMat拷贝了chromaticityMat，形成一个新的图像矩阵，两者相互没有影响
	namedWindow("对比：色度差检测结果",WINDOW_NORMAL);
	imshow("对比：色度差检测结果", chromaticityMat);
	waitKey(0);
	//色度差检测结果：背景为黄色，物体红色，阴影绿色。亮度差只需在阴影候选区中区分某像素是物体还是阴影
	int addObject=0;   //亮度比新检测出的物体像素数
	for(int i=0;i<brightnessMat.rows;i++)
	{
		for(int j=0;j<brightnessMat.cols;j++)
		{
			//亮度差只需在阴影候选区中区分某像素是物体还是阴影
			if( abs(chromaticityMat.at<Vec3b>(i,j)[0]-0)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[1]-255)==0 && abs(chromaticityMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//BD在阈值区间内，属于阴影
				//if( (bd_B[i][j]>bd_thresholdL_B && bd_B[i][j]<bd_thresholdH_B) || (bd_G[i][j]>bd_thresholdL_G && bd_G[i][j]<bd_thresholdH_G) || (bd_R[i][j]>bd_thresholdL_R && bd_R[i][j]<bd_thresholdH_R) )
				if( (bd_B[i][j]>bd_thresholdL_B && bd_B[i][j]<bd_thresholdH_B) && (bd_G[i][j]>bd_thresholdL_G && bd_G[i][j]<bd_thresholdH_G) && (bd_R[i][j]>bd_thresholdL_R && bd_R[i][j]<bd_thresholdH_R) )
				{
					continue;
				}  
				else
				{
					brightnessMat.at<Vec3b>(i,j)[0]=255;   //新检测的物体为蓝色
					brightnessMat.at<Vec3b>(i,j)[1]=0;
					brightnessMat.at<Vec3b>(i,j)[2]=0;

					addObject++;  //亮度比新检测出的物体像素数

					//修改结构体相应信息
					graph[i][j].category=1;
					graph[i][j].initColor_B=0;
					graph[i][j].initColor_G=0;
					graph[i][j].initColor_R=255;
				}
			}
		}
	}
	cout<<"亮度比新检测出的物体像素数："<<addObject<<endl;
	//测试struct存储的阴影信息是否与色度差检测结果一致
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{	
			if(graph[i][j].category==2)
				testNum++;
		}
	}
	cout<<"当前阴影像素总数为:"<<testNum<<endl;

	//此步显示是为了将亮度差检测结果与色度差检测结果进行对比
	namedWindow("对比：亮度差检测结果",WINDOW_NORMAL);
	imshow("对比：亮度差检测结果", brightnessMat);
	waitKey(0);
	destroyWindow("对比：色度差检测结果");
	destroyWindow("对比：亮度差检测结果");

	//保存对比图片
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.jpg", brightnessMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness_VS_chromaticity.bmp", brightnessMat);

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
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.jpg", brightnessMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.bmp", brightnessMat);

	return 0;
}

//局部亮度比
int localRelation()
{
	cout<<"-------------局部强度比检测阴影------------------"<<endl;
	//	localMat=brightnessMat.clone();   //深拷贝：localMat拷贝了brightnessMat，形成一个新的图像矩阵，两者相互没有影响
	localMat=imread("F:\\Code\\Shadow Detection\\Data\\Brightness Difference\\Brightness Differ Result\\20170228111043_brightness+chromaticity.bmp");  //读取图像
	namedWindow("色度+亮度差检测结果",WINDOW_NORMAL);
	imshow("色度+亮度差检测结果", localMat);
	waitKey(0);

	//初始化Q值
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			q_B[i][j]=0;
			q_G[i][j]=0;
			q_R[i][j]=0;
		}
	}

	//色度差+亮度差检测结果：背景为黄色，物体红色，阴影绿色。局部亮度比只需在阴影候选区中区分某像素是物体还是阴影
	//int sNum=0, notBoarder=0;
	for(int i=1;i<localMat.rows-1;i++)  //注：图片边缘无需计算Q值，注意i和j的取值范围
	{
		for(int j=1;j<localMat.cols-1;j++)
		{
			//局部亮度比只需在阴影候选区（绿色）中区分某像素是物体还是阴影
			if( abs(localMat.at<Vec3b>(i,j)[0]-0)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-0)==0 )
			{
				//sNum++;  //阴影像素个数

				//排除阴影边缘: 像素的邻域也要属于阴影
				if ( (localMat.at<Vec3b>(i,j-1)[0]==0 && localMat.at<Vec3b>(i,j-1)[1]==255 && localMat.at<Vec3b>(i,j-1)[2]==0) && (localMat.at<Vec3b>(i+1,j)[0]==0 && localMat.at<Vec3b>(i+1,j)[1]==255 && localMat.at<Vec3b>(i+1,j)[2]==0) && (localMat.at<Vec3b>(i,j+1)[0]==0 && localMat.at<Vec3b>(i,j+1)[1]==255 && localMat.at<Vec3b>(i,j+1)[2]==0) && (localMat.at<Vec3b>(i-1,j)[0]==0 && localMat.at<Vec3b>(i-1,j)[1]==255 && localMat.at<Vec3b>(i-1,j)[2]==0) )
				{
					q_B[i][j]= pow((bd_B[i][j-1]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i+1][j]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i][j+1]-bd_m_B)/bd_variance_B,2)+ pow((bd_B[i-1][j]-bd_m_B)/bd_variance_B,2);
					q_G[i][j]= pow((bd_G[i][j-1]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i+1][j]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i][j+1]-bd_m_G)/bd_variance_G,2)+ pow((bd_B[i-1][j]-bd_m_G)/bd_variance_G,2);
					q_R[i][j]= pow((bd_B[i][j-1]-bd_m_R)/bd_variance_R,2)+ pow((bd_B[i+1][j]-bd_m_R)/bd_variance_R,2)+ pow((bd_R[i][j+1]-bd_m_R)/bd_variance_R,2)+ pow((bd_B[i-1][j]-bd_m_R)/bd_variance_R,2);

					/*//输出Q值
					cout<<"q_B="<<q_B[i][j]<<"\t"<<"q_G="<<q_G[i][j]<<"\t"<<"q_R="<<q_R[i][j]<<endl;
					notBoarder++;  //非边缘阴影像素个数
					*/
				}
			}
		}
	}
	//cout<<"shadow Num:"<<sNum<<endl;
	//cout<<"shadow exclude boarder Num:"<<notBoarder<<endl;

	//利用Q值，将色度+亮度检测到的阴影，再次做判断
	int addObject=0;   //局部关系新检测出的物体像素数
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
						localMat.at<Vec3b>(i,j)[0]=255;   //新检测的物体为白色
						localMat.at<Vec3b>(i,j)[1]=255;
						localMat.at<Vec3b>(i,j)[2]=255;

						addObject++;  //局部关系新检测出的物体像素数

						//修改结构体相应信息
						graph[i][j].category=1;
						graph[i][j].initColor_B=0;
						graph[i][j].initColor_G=0;
						graph[i][j].initColor_R=255;
					}
				}
			}
		}
	}
	cout<<"局部关系新检测出的物体像素数："<<addObject<<endl;
	//测试struct存储的阴影信息是否与色度差检测结果一致
	int testNum=0;
	for(int i=0;i<HEIGHT;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{	
			if(graph[i][j].category==2)
				testNum++;
		}
	}
	cout<<"当前阴影像素总数为:"<<testNum<<endl;

	//此步显示是为了将局部对比检测结果与色度+亮度差检测结果进行对比
	namedWindow("局部对比检测结果",WINDOW_NORMAL);
	imshow("局部对比检测结果", localMat);
	waitKey(0);
	destroyWindow("局部对比检测结果");
	destroyWindow("色度+亮度差检测结果");

	//保存对比图片
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.jpg", localMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness_VS_local.bmp", localMat);

	//保存色度+亮度+局部对比的结果，统一颜色：阴影绿色，物体红色
	for(int i=0;i<localMat.rows;i++)
	{
		for(int j=0;j<localMat.cols;j++)
		{
			if( abs(localMat.at<Vec3b>(i,j)[0]-255)==0 && abs(localMat.at<Vec3b>(i,j)[1]-255)==0 && abs(localMat.at<Vec3b>(i,j)[2]-255)==0 )
			{
				localMat.at<Vec3b>(i,j)[0]=0;   	//将上次新检测到的物体颜色由白色改为红色即可
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
	//imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.jpg", localMat);
	imwrite("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp", localMat);

	return 0;
}

//利用连通域的包围关系优化阴影和物体
int spatialAjustment()
{
	cout<<"-------------利用连通域的包围关系优化结果------------------"<<endl;
	//spacialMat=localMat.clone();   //深拷贝：spacialMat拷贝了localMat，形成一个新的图像矩阵，两者相互没有影响
	spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp");  //读取图像
	//spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Color Space\\RGB\\Img_Rgb.jpg");
	namedWindow("对比：色度+亮度差+局部检测结果",WINDOW_NORMAL);
	imshow("对比：色度+亮度差+局部检测结果", spacialMat);
	waitKey(0);

	//将原图转为灰度图
	cvtColor(spacialMat, spacialGrayMat, CV_BGR2GRAY);

	//创建原图窗口并显示
	namedWindow(WINDOW_NAME1, CV_WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME1, spacialMat);
	waitKey(0);


	//创建滑动条来控制阈值
	/*  第一个参数：滑动条名称
	第二个参数：窗口名称
	第三个参数：当滑动条被拖到时，opencv会自动将当前位置所代表的值传递给指针指向的整数
	第四个参数：滑动条所能到达的最大值
	第五个参数：可选的回调函数,这里为自定义的阈值函数
	*/
	createTrackbar("阈值", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange);
	on_ThreshChange(0,0);   //初始化自定义的阈值函数

	//等待用户按键，如果是ESC，则退出等待过程
	while (true)
	{
		int c;
		c=waitKey(20);
		if((char)c==27)
			break;
	}

	return 0;
}

//自定义的阈值函数
void on_ThreshChange(int, void*)
{
	Mat src_copy=spacialMat.clone();
	Mat threshold_output;
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//对图像进行二值化
	/*  threshold函数：遍历灰度图，将图像信息二值化，处理过后的图片只有两种色值
	第一个参数：输入，必须为单通道，8bit或32bit浮点类型的Mat即可
	第二个参数：存放输出结果，且与第一个参数有相同的尺寸和类型
	第三个参数：阈值的具体值
	第四个参数：maxvalue，当第五个参数取THRESH_BINARY或THRESH_BINARY_INV类型时的最大值（二值化：0黑，255白）
	第五个参数：阈值类型：THRESH_BINARY 当前点大于阈值时，取maxvalue（即第四个参数），否则设置为0
	*/
	threshold(spacialGrayMat, threshold_output, g_nThresh, 255, THRESH_BINARY);

	
	//寻找轮廓
	/*  第一个参数：输入图像，8bit的单通道二值图像
	contours：检测到的轮廓，是一个向量，每个元素都是一个轮廓。因此，这个向量的每个元素都是一个向量，即vector<vector<Point>>contours
	hierarchy:各个轮廓的继承关系。hierarchy也是一个向量，长度与contours相等，每个元素和contours的元素对应。
	hierarchy的每个元素是一个包含四个整型数的向量，即vector<Vec4i>hierarchy
	hierarchy[i][0],hierarchy[i][1],hierarchy[i][2],hierarchy[i][3]分别表示第i条轮廓（contours[i])的下一条，前一条，包含的第一条子轮廓和包含它的父轮廓
	第四个参数：检测轮廓的方法，共有四种。CV_RETR_TREE检测所有轮廓，并建立所有的继承（包含）关系。
	第五个参数：表示一条轮廓的方法。CV_CHAIN_APPROX_SIMPLE只存储水平、垂直、对角直线的起始点。
	第六个参数：每一个轮廓点的偏移量，当轮廓是从图形ROI中（感兴趣区）提取出来的时候，使用偏移量有用，因为可以从整个图像上下文来对轮廓做分析
	例如，想从图像的(100,0)开始进行轮廓检测，就传入（100，0）
	*/
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	//对每个轮廓计算其凸包
	//凸包：一个轮廓可以有无数个包围它的外壳，其中表面积最小的外壳，就是凸包
	vector<vector<Point>>hull(contours.size());
	for(int i=0; i<contours.size(); i++)
	{
		/*  第一个参数：要求的凸包的点集
		第二个参数：输出的凸包点
		第三个参数：bool变量，表示求得的凸包是顺时针还是逆时针方向。true是顺时针
		*/
		convexHull(Mat(contours[i]), hull[i], false);
	}

	//绘出轮廓及其凸包
	Mat drawing=Mat::zeros(threshold_output.size(), CV_8UC3);   //返回指定大小和类型的零数组
	for(int i=0; i<contours.size(); i++)
	{
		//scalar：定义可存放1--4个数值的数组
		//uniform：返回指定范围的随机数
		Scalar color=Scalar(g_rng.uniform(0,255), g_rng.uniform(0,255), g_rng.uniform(0,255));

		/*drawContours：画出图像的轮廓
		第一个参数：目标图像
		第二个参数：输入的轮廓组，每一组轮廓由点vector构成
		第三个参数：指明画第几个轮廓
		第四个参数：轮廓的颜色
		第五个参数：轮廓的线宽。如果为负值或者CV_FILLED表示填充轮廓内部
		第六个参数：线条的类型
		第七个参数：轮廓结构信息
		第八个参数：MAX_LEVEL，绘制轮廓的最大等级。如果为0，绘制单独的轮廓；如果为1，绘制轮廓及其后的相同级别的轮廓。如果为2，所有的轮廓。
		第九个参数：按照给出的偏移量移动每一个轮廓点坐标。当轮廓是从某些感兴趣区域（ROI）中提取时，需要考虑ROI偏移量，会用到这个参数
		*/
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		/*		//计算每个轮廓的面积
		double area = fabs(contourArea(contours[i], true));
		cout<<"第"<<i<<"个轮廓的面积为："<<area<<endl;
		*/
	}

	//把结果显示在窗体
	namedWindow(WINDOW_NAME2, CV_WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME2, drawing);
}

//填充小连通域
void fillSmallDomain()
{
	spacialMat=imread("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp");  //读取图像
	//统计当前的物体(红色）像素个数
	int objectNum=0;
	int shadowNum=0;
	int backNum=0;
	for(int i=0;i<spacialMat.rows;i++)
	{
		for(int j=0;j<spacialMat.cols;j++)
		{
			graph[i][j].revise=0;
			//物体：红色
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
	cout<<"当前物体像素个数："<<objectNum<<endl;
	cout<<"当前阴影像素个数："<<shadowNum<<endl;
	cout<<"当前背景像素个数："<<backNum<<endl;


	//定义小连通域：物体像素总数的4%
	double connectedDomain;
	connectedDomain = objectNum * 0.04;
	cout<<"连通域包含的像素个数："<<connectedDomain<<endl;

	//-------------基于之前的阈值调整，以下利用最优的阈值------------------------------
	IplImage* src=NULL;
	IplImage* img=NULL;
	IplImage* dst=NULL;

	CvMemStorage* storage=cvCreateMemStorage(0);
	CvSeq* contour=0;

	CvScalar external_color;  //外轮廓颜色。图像二值化后，只有黑色和白色，白色区域的轮廓是“外轮廓”
	CvScalar hole_color;  //内轮廓颜色，黑色区域的轮廓是“内轮廓”

	src=cvLoadImage("F:\\Code\\Shadow Detection\\Data\\Local Relation\\Local Relation Result\\20170228111043_brightness+local.bmp",1);
	img=cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	dst=cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

	cvCvtColor(src, img, CV_BGR2GRAY);
	//注意！！！！！此处的100是基于之前手动阈值调整自己设置的最优寻找轮廓的阈值！！！！
	cvThreshold(img, img, 100, 200, CV_THRESH_BINARY);

	//找到二值图像中的轮廓
	/*  CV_RETR_LIST：提取所有轮廓，并放置在list中
	CV_CHAIN_APPROX_NONE：将所有点由链码形式转化为点序列形式
	*/
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cvZero(dst);  //清空数组

	//_contour：为了保存轮廓的首指针位置，因为随后contour将用来迭代
	CvSeq* _contour=contour;

	//----------------------画外轮廓和内轮廓-------------
	IplImage* test=NULL;
	test=cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	CvScalar colorEx1=CV_RGB(255, 0, 0);  //外层轮廓红色
	CvScalar colorEx2=CV_RGB(0, 255, 0);  //内层轮廓绿色

	CvSeq* cNext=NULL;
	bool first=true;
	int count=0;  //轮廓数量
	int tempRow,tempCol;
	int tempB,tempG,tempR;
	int up=0,right=0,left=0,down=0,none=0,upleft=0,upright=0,downleft=0,downright=0;  //八邻域
	for(CvSeq* c=contour; c!=NULL; c=cNext)
	{
		double tmp=fabs(cvContourArea(c));  //计算轮廓面积
		//------------删除大连通域------------
		if(tmp>connectedDomain)  //面积大于最小连通域，删除
		{
			//删除大面积
			cNext=c->h_next;
			cvClearSeq(c);
			continue;
		}
		else
		{
			if(first)  //如果序列中的第一个轮廓被删除，则将序列首指针指向它的下一个元素
				contour=c;
			first=false;

			//**************填充小轮廓*****************
			//c->total：序列contours_tmp中点的总数
			for(int j=0; j<c->total; j++)  //提取一个轮廓的所有坐标点
			{
				//cvGetSeqElem（）：返回索引指定的元素指针
				CvPoint *pt=(CvPoint*)cvGetSeqElem(c, j);  //cvGetSeqElem：得到轮廓中的一个点

				//存储该位置像素的位置信息
				tempRow=pt->y;
				tempCol=pt->x;
				//排除图像边缘
				if(tempRow!=0 || tempRow!=HEIGHT-1 || tempCol!=0 || tempCol!=WIDTH-1)
				{
					//存储该位置像素的颜色信息
					tempB=spacialMat.at<Vec3b>(tempRow,tempCol)[0];
					tempG=spacialMat.at<Vec3b>(tempRow,tempCol)[1];
					tempR=spacialMat.at<Vec3b>(tempRow,tempCol)[2];

					//该像素八邻域的颜色信息
					//当邻域像素与该位置颜色不同时，就不必继续看其他邻域像素，可以直接对该轮廓进行填充
					int b,g,r;
					//上边的点
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						// cvDrawContours()：在图像上绘制外部内部轮廓
						//第一个参数：要在其上绘制轮廓的图像
						//第二个参数：指向第一个轮廓的指针
						//第三个参数：外轮廓的颜色
						//第四个参数：内轮廓的颜色
						//第五个参数：画轮廓的最大层数。0：只绘制contours_tmp
						//第六个参数：绘制轮廓线的宽度。CV_FILLED：contours_tmp内部将被绘制
						//第七个参数：轮廓线段的类型
						//第八个参数：按给定值移动所有点的坐标
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						up++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影
							
						break;
					}

					//右边的点
					b=spacialMat.at<Vec3b>(tempRow,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						right++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//下边的点
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						down++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//左边的点
					b=spacialMat.at<Vec3b>(tempRow,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						left++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//左上边的点
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						upleft++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//右上边的点
					b=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow-1,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						upright++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//左下边的点
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol-1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						downleft++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

						break;
					}

					//右下边的点
					b=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[0];
					g=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[1];
					r=spacialMat.at<Vec3b>(tempRow+1,tempCol+1)[2];
					if(b!=tempB && g!=tempG && r!=tempR)
					{
						cvDrawContours(test, c, CV_RGB(r,g,b), CV_RGB(r,g,b), 0, CV_FILLED, 8, cvPoint(0,0));
						downright++;

						//修改像素的信息
						graph[tempRow][tempCol].initColor_B=b;
						graph[tempRow][tempCol].initColor_G=g;
						graph[tempRow][tempCol].initColor_R=r;
						if(b==0 && g==255 && r==255)
						{	
							graph[tempRow][tempCol].category=0;  //背景
						}
						else if(b==0 && g==0 && r==255)
						{
							graph[tempRow][tempCol].category=1;  //物体
						}
						else
							graph[tempRow][tempCol].category=2;  //阴影

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
	cout<<"小连通域个数："<<count<<endl;
	cout<<"up个数："<<up<<endl;
	cout<<"right个数："<<right<<endl;
	cout<<"down个数："<<down<<endl;
	cout<<"left个数："<<left<<endl;
	cout<<"upleft个数："<<upleft<<endl;
	cout<<"upright个数："<<upright<<endl;
	cout<<"downleft个数："<<downleft<<endl;
	cout<<"downright个数："<<downright<<endl;
	cout<<"none个数："<<none<<endl;
	cvNamedWindow("fill image", CV_WINDOW_AUTOSIZE);
	cvShowImage("fill image", test);
	cvWaitKey(0);


	//------------保存新的图像-------
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
	namedWindow("填充最小连通域",WINDOW_NORMAL);
	imshow("填充最小连通域", spacialMat);
	waitKey(0);
	destroyWindow("填充最小连通域");
	cvDestroyWindow("filter image");	
	cvDestroyWindow("fill image");
	cvReleaseImage(&test);
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvReleaseMemStorage(&storage);

	//保存进一步检测的图片
	imwrite("F:\\Code\\Shadow Detection\\Data\\Spacial Improved\\20170228111043_brightness+local+spacial.bmp", spacialMat);
}


//阴影检测算法
int shadowDetection()
{
	//step1. 色度差阴影检测
//	chromaticityDiffer();

	//step2. 亮度差阴影检测
	//注：这个函数要用到chromaticityDiffer()
//	brightnessDiffer();

	//step3. 局部亮度比
//	localRelation();

	//step4.利用连通域的包围关系优化阴影和物体
//	spatialAjustment();  //手动选取阈值

	//step5.填充最小连通域
	fillSmallDomain();   //填充最小连通域

	return 0;
}