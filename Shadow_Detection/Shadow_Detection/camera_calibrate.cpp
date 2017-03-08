/*
------------------------------------------------
Author: CIEL
Date: 2017/02/13
Function: 张正友法相机标定
------------------------------------------------
*/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "camera_calibrate.h"

using namespace cv;
using namespace std;

int camera_calibrate()
{
//	ifstream fin("G:\\Code-Shadow Detection\\Data\\Camera Calibrate\\calibdata.txt");  //标定所用图像文件的路径,ifstram文件读操作
	
//	ifstream fin("g:\\Code-Shadow Detection\\Data\\example camera calibration\\calibdata.txt");
	ifstream fin("calibdata.txt");
//	ofstream fout("g:\\Code-Shadow Detection\\Data\\Camera Calibrate\\result\\caliberation_result.txt");  //保存标定结果的文件,ofstream文件写操作
	ofstream fout("caliberation_result.txt");  //保存标定结果的文件,ofstream文件写操作
	//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
	cout<<"-------开始提取角点···-------";
	int image_cout=0;  //图像数量
	Size image_size;  //图像尺寸
	Size board_size=Size(7,7);   //标定板上每行、列的角点数
//	Size board_size=Size(4,6); 
	vector<Point2f> image_points_buf;  //缓存每幅图像上检测到的角点
	vector<vector<Point2f>> image_points_seq;   //保存检测到的所有角点
	string filename;
	int count=-1;   //用于存储角点个数
	while(getline(fin,filename))  //按行读取文件
	{
		image_cout++;
		//用于观察检验输出
		cout<<"image_cout="<<image_cout<<endl;
		//输出检验
		cout<<"-->cout="<<count;

		Mat imageInput=imread(filename);
		if(image_cout==1) //读入第一张图片时获取的图像宽、高信息
		{
			image_size.width=imageInput.cols;
			image_size.height=imageInput.rows;
			cout<<"image_size.width="<<image_size.width<<endl;
			cout<<"image_size.height="<<image_size.height<<endl;
		}

		//提取角点
		/*findChessboardCorners()提取的角点专指标定板上的内角点，这些角点与标定板的边缘不接触
		第一个参数：传入拍摄的棋盘图Mat图像，必须是8位的灰度或彩色图像
		第二个参数：每个棋盘图上内角点的行列数，一般情况下，行列数不要相同
		第三个参数：用于存储检测到的内角点图像坐标，一般用元素point2f的向量来表示
		第四个参数：用于定义棋盘图上内角点查找的不同处理方式，有默认值
		*/
		if(0== findChessboardCorners(imageInput, board_size, image_points_buf) )
		{
			cout<<"can not find chessboard corners!"<<endl;
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  //将图像从RGB空间转到灰度图
			/*亚像素精确化
			为提高标定精度，需要在初步提取的角点信息上进一步提取亚像素信息，降低相机标定偏差
			常用方法：法1find4QuadCornerSubpix，法2cornerSubPix，两种方法有一定差距，但偏差基本控制在0.5个像素以内
			*/
			/*	法1   find4QuadCornerSubpix：
			第一个参数：输入的图像Mat矩阵，最好是8位灰度图，检测效率更高
			第二个参数：初始角点的坐标，同时作为亚像素坐标位置的输出，所以需要是浮点型，一般用point2f或point2d的向量来表示
			第三个参数：角点搜索窗口的尺寸
			*/
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5,5) );  //对粗提取的角点进行精确化
			/*  法2    cornerSubPix
			第一个参数：输入的图像Mat矩阵，最好是8位灰度图，检测效率更高
			第二个参数：初始角点的坐标，同时作为亚像素坐标位置的输出，所以需要是浮点型，一般用point2f或point2d的向量来表示
			第三个参数：大小为搜索窗口的一半
			第四个参数：死区的一半尺寸，当值为（-1，-1）时表示没有死区。死区为对不搜索区的中央位置做求和运算的区域，用来避免自相关矩阵出现某些可能的奇异性。
			第五个参数：定义求角点的迭代过程的终止条件，可以为迭代次数和角点精确度两者的组合。
			*/
			//cornerSubPix(view_gray, image_points_buf, Size(5,5), Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
			
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点

			//在图像上显示角点位置,非必须，仅为了显示
			/*drawChessboardCorners():用于绘制被成功标定的角点
			第一个参数：图像，灰度图或彩图
			第二个参数：每张标定棋盘上内角点的行列数
			第三个参数：初始角点坐标向量，同时作为亚像素坐标位置的输出，所以需要是浮点型，一般用point2f或point2d的向量来表示
			第四个参数：标志位，用来定义棋盘内角点是否被完整的探测到，true表示完整的探测到，函数会用直线一次连接所有内角点，作为一个整体；false表示有未被探测到的内角点，函数会以圆圈标记处检测到的内角点
			*/
			drawChessboardCorners(view_gray, board_size, image_points_buf, false);  //用于在图片中标记角点
			imshow("Camera Calibration", view_gray);  //显示图片
			waitKey(500);   //暂停0.55
		}
	}

	int total=image_points_seq.size();
	cout<<"total="<<total<<endl;
	int CornerNum=board_size.width*board_size.height;  //每张图片上的角点数
	for(int ii=0; ii<total; ii++)
	{
		if(0== ii%CornerNum)  //此判断语句是为了输出图片号，便于控制台观看
		{
			int i=-1;
			i=ii/CornerNum;
			int j=i+1;  //j为图片号
			cout<<"-->第"<<j<<"图片的数据-->:"<<endl;
		}
		if(0==ii%3)  //此判断语句，格式化输出，便于控制台观看
		{
			cout<<endl;
		}
		else
		{
			cout.width(10);
		}
		//输出所有角点
		cout<<"-->"<<image_points_seq[ii][0].x;
		cout<<"-->"<<image_points_seq[ii][0].y;
	}
	cout<<"角点提取完成！"<<endl;

	//相机标定
	cout<<"------ 开始标定···-------";
	//棋盘三维信息
	Size square_size=Size(16,16);  //实际测量得到的标定板上每个棋盘格的大小
//	Size square_size=Size(10,10);
	vector<vector<Point3f>> object_points;  //保存标定板上角点的三维坐标
	//相机内外参数
	Mat cameraMatrix=Mat(3, 3, CV_32FC1, Scalar::all(0) );  //内参矩阵，Mat(int _rows, int _cols, int _type, const Scalar& _s)
	vector<int> point_counts;  //每幅图像中角点的数量
	Mat distCoeffs=Mat(1, 5, CV_32FC1, Scalar::all(0));  //相机的5个畸变系数：k1, k2, p1, p2, k3
	vector<Mat> tvecsMat; //每幅图像的旋转向量
	vector<Mat> rvecsMat;  //每幅图像的平移向量

	//初始化标定板上角点的三维坐标
	int i, j, t;
	for(t=0; t<image_cout; t++)
	{
		vector<Point3f> tempPointSet;
		for(i=0; i<board_size.height; i++)
		{
			for(j=0; j<board_size.width; j++)
			{
				Point3f realPoint;
				//假设标定板放在世界坐标系中z=0的平面上
				realPoint.x=i*square_size.width;
				realPoint.y=j*square_size.height;
				realPoint.z=0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	//初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
	for(i=0; i<image_cout; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	//开始标定
	/*
	calibrateCamera函数
	第一个参数：世界坐标系中的三维点。使用时，应输入一个三维坐标点的向量的向量，即vector<vector<Point3f>> object_points。需要依据棋盘上单个黑白矩阵的大小，计算出（初始化）每一个内角点的世界坐标
	第二个参数：每个内角点对应的图像坐标点
	第三个参数：图像的像素尺寸大小。在计算相机的内参和畸变矩阵时需要使用到该参数
	第四个参数：相机的内参矩阵
	第五个参数：畸变矩阵
	第六个参数：旋转向量
	第七个参数：平移向量
	第八个参数：标定时所采用的算法。
	在使用该函数前，需要对棋盘上每一个内角点的空间坐标系的位置坐标进行初始化，
	标定的结果：生成相机的内参矩阵、5个畸变系数，每个图像会生成属于自己的平移向量和旋转向量
	*/
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	
	cout<<"标定完成！"<<endl;

	//对标定结果进行评价
	//评价方法：通过得到的相机内外参数，对空间的三维点进行重新投影计算，得到在图片上新的投影点的坐标。计算投影坐标和亚像素角点坐标之间的偏差，偏差越小越好。
	cout<<"------- 开始评价标定结果··· --------"<<endl;
	double total_err=0.0;  //所有图像的平均误差的总和
	double err=0.0;  //每幅图像的平均误差
	vector<Point2f> image_points2;  //保存重新计算得到的投影点
	cout<<"\t每幅图像的标定误差："<<endl;
	fout<<"每幅图像的标定误差:"<<endl;
	for(i=0; i<image_cout; i++)
	{
		vector<Point3f> tempPointSet=object_points[i];
		//通过得到的相机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
		/*
		projectPoints:对空间三维坐标点进行反向投影的函数
		第一个参数：相机坐标系中的三维点坐标
		第二个参数：旋转向量。每个图像都有自己的旋转向量
		第三个参数：平移向量。每个图像都有自己的平移向量
		第四个参数：相机的内参矩阵
		第五个参数：畸变矩阵
		第六个参数：每个内角点对应的图像上的坐标点
		*/
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

		//计算新的投影点和旧的投影点之间的误差
		vector<Point2f> tempImagePoint=image_points_seq[i];
		Mat tempImagePointMat=Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat=Mat(1, image_points2.size(), CV_32FC2);
		for(int j=0; j<tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0,j)=Vec2f(image_points2[j].x, image_points2[j].y);  //Mat::at()，返回指定数组元素的引用
			tempImagePointMat.at<Vec2f>(0,j)=Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err=norm(image_points2Mat, tempImagePointMat, NORM_L2);  //计算两个数组的第二范数
		total_err += err/= point_counts[i];
		std::cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
		fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<endl;
	}
	std::cout<<"总体平均误差："<<total_err/image_cout<<"像素"<<endl;
	fout<<"总体平均误差："<<total_err/image_cout<<"像素"<<endl;
	std::cout<<"评价完成!"<<endl;

	//保存标定结果
	std::cout<<"------- 开始保存标定结果··· --------"<<endl;
	Mat ratation_matrix=Mat(3, 3, CV_32FC1, Scalar::all(0) );  //保存每幅图像的旋转矩阵
	fout<<"相机的内参数矩阵:"<<endl;
	fout<<cameraMatrix<<endl<<endl;
	fout<<"相机的畸变系数:"<<endl;
	fout<<distCoeffs<<endl<<endl<<endl;
	for(int i=0; i<image_cout; i++)
	{
		fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;
		fout<<tvecsMat[i]<<endl;

		//将旋转向量转换为对应的旋转矩阵:罗德里格斯（Rodrigues）变换
		//一个向量乘以旋转矩阵等价于向量以某种方式进行旋转；旋转向量的长度（模）表示绕轴逆时针旋转的角度（弧度）
		Rodrigues(tvecsMat[i], ratation_matrix);
		fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;
		fout<<ratation_matrix<<endl;
		fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;
		fout<<rvecsMat[i]<<endl;
	}
	std::cout<<"完成保存"<<endl;
	fout<<endl;

	//显示标定结果:利用求得的相机内参和外参，对图像进行畸变的矫正
	Mat mapx=Mat(image_size, CV_32FC1);
	Mat mapy=Mat(image_size, CV_32FC1);
	Mat R=Mat::eye(3, 3, CV_32F);  //Mat::eye()，返回一个恒等指定大小和类型的矩阵
	std::cout<<"保存矫正图像"<<endl;
	string imageFilename;
	std::stringstream StrStm;
	for(int i=0; i!=image_cout; i++)
	{
		std::cout<<"Frame #"<<i+1<<"···"<<endl;

		//法1：	用initUndistortRectifyMap和remap两个函数配合，initUndistortRectifyMap计算畸变映射，remap把求得的映射应用到图像上
		/*
		initUndistortRectifyMap计算畸变映射
		第一个参数：之前求得的相机内参矩阵
		第二个参数：之前求得的相机畸变矩阵
		第三个参数：可选择的输入，是第一和第二相机坐标之间的旋转矩阵
		第四个参数：输入的矫正后的3*3相机矩阵
		第五个参数：相机采集的无失真的图像尺寸
		第六个参数：定义第七和第八个参数的数据类型
		第七个和第八个参数：输出的X/Y坐标重映射参数
		*/
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		StrStm.clear();
//		string filePath="G:\\Code-Shadow Detection\\Data\\Camera Calibrate\\chess";
//		string filePath="G:\\Code-Shadow Detection\\Data\\example camera calibration\\chess";
		string filePath="chess";
		StrStm<<i+1;
		StrStm>>imageFilename;
		filePath += imageFilename;
		filePath += ".jpg";
//		filePath += ".bmp";
		Mat imageSource=imread(filePath);
		Mat newimage=imageSource.clone();
		/*
		remap把求得的映射应用到图像上
		第一个参数：输入畸变的原始图像
		第二个参数：矫正后输出的图像，与输入的图像有相同的类型和大小
		第三和第四个参数：X/Y坐标的映射
		第五个参数：定义图像的插值方式
		第六个参数：定义边界填充方式
		*/
		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);

		//法2：undistort函数实现
		/*第一个参数：输入畸变的原始图像
		第二个参数：矫正后输出的图像，与输入的图像有相同的类型和大小
		第三个参数：之前求得的相机内参矩阵
		第四个参数：之前求得的相机畸变矩阵*/
		//法1比法2效率更高
		//undistort(imageSource, newimage, cameraMatrix, distCoeffs);

		StrStm.clear();
		filePath.clear();

//		string resultfilePath="G:\\Code-Shadow Detection\\Data\\Camera Calibrate\\result\\chess";
//		string resultfilePath="g:\\Code-Shadow Detection\\Data\\example camera calibration\\result\\chess";
		string resultfilePath="chess";
		StrStm<<i+1;
		StrStm>>imageFilename;
		resultfilePath += imageFilename;
		resultfilePath += "_calibration.jpg";


		imwrite(resultfilePath, newimage);
	}
	std::cout<<"保存结果"<<endl;
	

	return 0;
}
 