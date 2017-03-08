/*
------------------------------------------------
Author: CIEL
Date: 2017/02/13
Function: �����ѷ�����궨
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
//	ifstream fin("G:\\Code-Shadow Detection\\Data\\Camera Calibrate\\calibdata.txt");  //�궨����ͼ���ļ���·��,ifstram�ļ�������
	
//	ifstream fin("g:\\Code-Shadow Detection\\Data\\example camera calibration\\calibdata.txt");
	ifstream fin("calibdata.txt");
//	ofstream fout("g:\\Code-Shadow Detection\\Data\\Camera Calibrate\\result\\caliberation_result.txt");  //����궨������ļ�,ofstream�ļ�д����
	ofstream fout("caliberation_result.txt");  //����궨������ļ�,ofstream�ļ�д����
	//��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
	cout<<"-------��ʼ��ȡ�ǵ㡤����-------";
	int image_cout=0;  //ͼ������
	Size image_size;  //ͼ��ߴ�
	Size board_size=Size(7,7);   //�궨����ÿ�С��еĽǵ���
//	Size board_size=Size(4,6); 
	vector<Point2f> image_points_buf;  //����ÿ��ͼ���ϼ�⵽�Ľǵ�
	vector<vector<Point2f>> image_points_seq;   //�����⵽�����нǵ�
	string filename;
	int count=-1;   //���ڴ洢�ǵ����
	while(getline(fin,filename))  //���ж�ȡ�ļ�
	{
		image_cout++;
		//���ڹ۲�������
		cout<<"image_cout="<<image_cout<<endl;
		//�������
		cout<<"-->cout="<<count;

		Mat imageInput=imread(filename);
		if(image_cout==1) //�����һ��ͼƬʱ��ȡ��ͼ�������Ϣ
		{
			image_size.width=imageInput.cols;
			image_size.height=imageInput.rows;
			cout<<"image_size.width="<<image_size.width<<endl;
			cout<<"image_size.height="<<image_size.height<<endl;
		}

		//��ȡ�ǵ�
		/*findChessboardCorners()��ȡ�Ľǵ�רָ�궨���ϵ��ڽǵ㣬��Щ�ǵ���궨��ı�Ե���Ӵ�
		��һ���������������������ͼMatͼ�񣬱�����8λ�ĻҶȻ��ɫͼ��
		�ڶ���������ÿ������ͼ���ڽǵ����������һ������£���������Ҫ��ͬ
		���������������ڴ洢��⵽���ڽǵ�ͼ�����꣬һ����Ԫ��point2f����������ʾ
		���ĸ����������ڶ�������ͼ���ڽǵ���ҵĲ�ͬ����ʽ����Ĭ��ֵ
		*/
		if(0== findChessboardCorners(imageInput, board_size, image_points_buf) )
		{
			cout<<"can not find chessboard corners!"<<endl;
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  //��ͼ���RGB�ռ�ת���Ҷ�ͼ
			/*�����ؾ�ȷ��
			Ϊ��߱궨���ȣ���Ҫ�ڳ�����ȡ�Ľǵ���Ϣ�Ͻ�һ����ȡ��������Ϣ����������궨ƫ��
			���÷�������1find4QuadCornerSubpix����2cornerSubPix�����ַ�����һ����࣬��ƫ�����������0.5����������
			*/
			/*	��1   find4QuadCornerSubpix��
			��һ�������������ͼ��Mat���������8λ�Ҷ�ͼ�����Ч�ʸ���
			�ڶ�����������ʼ�ǵ�����꣬ͬʱ��Ϊ����������λ�õ������������Ҫ�Ǹ����ͣ�һ����point2f��point2d����������ʾ
			�������������ǵ��������ڵĳߴ�
			*/
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5,5) );  //�Դ���ȡ�Ľǵ���о�ȷ��
			/*  ��2    cornerSubPix
			��һ�������������ͼ��Mat���������8λ�Ҷ�ͼ�����Ч�ʸ���
			�ڶ�����������ʼ�ǵ�����꣬ͬʱ��Ϊ����������λ�õ������������Ҫ�Ǹ����ͣ�һ����point2f��point2d����������ʾ
			��������������СΪ�������ڵ�һ��
			���ĸ�������������һ��ߴ磬��ֵΪ��-1��-1��ʱ��ʾû������������Ϊ�Բ�������������λ����������������������������ؾ������ĳЩ���ܵ������ԡ�
			�����������������ǵ�ĵ������̵���ֹ����������Ϊ���������ͽǵ㾫ȷ�����ߵ���ϡ�
			*/
			//cornerSubPix(view_gray, image_points_buf, Size(5,5), Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
			
			image_points_seq.push_back(image_points_buf);  //���������ؽǵ�

			//��ͼ������ʾ�ǵ�λ��,�Ǳ��룬��Ϊ����ʾ
			/*drawChessboardCorners():���ڻ��Ʊ��ɹ��궨�Ľǵ�
			��һ��������ͼ�񣬻Ҷ�ͼ���ͼ
			�ڶ���������ÿ�ű궨�������ڽǵ��������
			��������������ʼ�ǵ�����������ͬʱ��Ϊ����������λ�õ������������Ҫ�Ǹ����ͣ�һ����point2f��point2d����������ʾ
			���ĸ���������־λ���������������ڽǵ��Ƿ�������̽�⵽��true��ʾ������̽�⵽����������ֱ��һ�����������ڽǵ㣬��Ϊһ�����壻false��ʾ��δ��̽�⵽���ڽǵ㣬��������ԲȦ��Ǵ���⵽���ڽǵ�
			*/
			drawChessboardCorners(view_gray, board_size, image_points_buf, false);  //������ͼƬ�б�ǽǵ�
			imshow("Camera Calibration", view_gray);  //��ʾͼƬ
			waitKey(500);   //��ͣ0.55
		}
	}

	int total=image_points_seq.size();
	cout<<"total="<<total<<endl;
	int CornerNum=board_size.width*board_size.height;  //ÿ��ͼƬ�ϵĽǵ���
	for(int ii=0; ii<total; ii++)
	{
		if(0== ii%CornerNum)  //���ж������Ϊ�����ͼƬ�ţ����ڿ���̨�ۿ�
		{
			int i=-1;
			i=ii/CornerNum;
			int j=i+1;  //jΪͼƬ��
			cout<<"-->��"<<j<<"ͼƬ������-->:"<<endl;
		}
		if(0==ii%3)  //���ж���䣬��ʽ����������ڿ���̨�ۿ�
		{
			cout<<endl;
		}
		else
		{
			cout.width(10);
		}
		//������нǵ�
		cout<<"-->"<<image_points_seq[ii][0].x;
		cout<<"-->"<<image_points_seq[ii][0].y;
	}
	cout<<"�ǵ���ȡ��ɣ�"<<endl;

	//����궨
	cout<<"------ ��ʼ�궨������-------";
	//������ά��Ϣ
	Size square_size=Size(16,16);  //ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С
//	Size square_size=Size(10,10);
	vector<vector<Point3f>> object_points;  //����궨���Ͻǵ����ά����
	//����������
	Mat cameraMatrix=Mat(3, 3, CV_32FC1, Scalar::all(0) );  //�ڲξ���Mat(int _rows, int _cols, int _type, const Scalar& _s)
	vector<int> point_counts;  //ÿ��ͼ���нǵ������
	Mat distCoeffs=Mat(1, 5, CV_32FC1, Scalar::all(0));  //�����5������ϵ����k1, k2, p1, p2, k3
	vector<Mat> tvecsMat; //ÿ��ͼ�����ת����
	vector<Mat> rvecsMat;  //ÿ��ͼ���ƽ������

	//��ʼ���궨���Ͻǵ����ά����
	int i, j, t;
	for(t=0; t<image_cout; t++)
	{
		vector<Point3f> tempPointSet;
		for(i=0; i<board_size.height; i++)
		{
			for(j=0; j<board_size.width; j++)
			{
				Point3f realPoint;
				//����궨�������������ϵ��z=0��ƽ����
				realPoint.x=i*square_size.width;
				realPoint.y=j*square_size.height;
				realPoint.z=0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	//��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨��
	for(i=0; i<image_cout; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	//��ʼ�궨
	/*
	calibrateCamera����
	��һ����������������ϵ�е���ά�㡣ʹ��ʱ��Ӧ����һ����ά��������������������vector<vector<Point3f>> object_points����Ҫ���������ϵ����ڰ׾���Ĵ�С�����������ʼ����ÿһ���ڽǵ����������
	�ڶ���������ÿ���ڽǵ��Ӧ��ͼ�������
	������������ͼ������سߴ��С���ڼ���������ڲκͻ������ʱ��Ҫʹ�õ��ò���
	���ĸ�������������ڲξ���
	������������������
	��������������ת����
	���߸�������ƽ������
	�ڰ˸��������궨ʱ�����õ��㷨��
	��ʹ�øú���ǰ����Ҫ��������ÿһ���ڽǵ�Ŀռ�����ϵ��λ��������г�ʼ����
	�궨�Ľ��������������ڲξ���5������ϵ����ÿ��ͼ������������Լ���ƽ����������ת����
	*/
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	
	cout<<"�궨��ɣ�"<<endl;

	//�Ա궨�����������
	//���۷�����ͨ���õ����������������Կռ����ά���������ͶӰ���㣬�õ���ͼƬ���µ�ͶӰ������ꡣ����ͶӰ����������ؽǵ�����֮���ƫ�ƫ��ԽСԽ�á�
	cout<<"------- ��ʼ���۱궨��������� --------"<<endl;
	double total_err=0.0;  //����ͼ���ƽ�������ܺ�
	double err=0.0;  //ÿ��ͼ���ƽ�����
	vector<Point2f> image_points2;  //�������¼���õ���ͶӰ��
	cout<<"\tÿ��ͼ��ı궨��"<<endl;
	fout<<"ÿ��ͼ��ı궨���:"<<endl;
	for(i=0; i<image_cout; i++)
	{
		vector<Point3f> tempPointSet=object_points[i];
		//ͨ���õ����������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ��
		/*
		projectPoints:�Կռ���ά�������з���ͶӰ�ĺ���
		��һ���������������ϵ�е���ά������
		�ڶ�����������ת������ÿ��ͼ�����Լ�����ת����
		������������ƽ��������ÿ��ͼ�����Լ���ƽ������
		���ĸ�������������ڲξ���
		������������������
		������������ÿ���ڽǵ��Ӧ��ͼ���ϵ������
		*/
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

		//�����µ�ͶӰ��;ɵ�ͶӰ��֮������
		vector<Point2f> tempImagePoint=image_points_seq[i];
		Mat tempImagePointMat=Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat=Mat(1, image_points2.size(), CV_32FC2);
		for(int j=0; j<tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0,j)=Vec2f(image_points2[j].x, image_points2[j].y);  //Mat::at()������ָ������Ԫ�ص�����
			tempImagePointMat.at<Vec2f>(0,j)=Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err=norm(image_points2Mat, tempImagePointMat, NORM_L2);  //������������ĵڶ�����
		total_err += err/= point_counts[i];
		std::cout<<"��"<<i+1<<"��ͼ���ƽ����"<<err<<"����"<<endl;
		fout<<"��"<<i+1<<"��ͼ���ƽ����"<<err<<endl;
	}
	std::cout<<"����ƽ����"<<total_err/image_cout<<"����"<<endl;
	fout<<"����ƽ����"<<total_err/image_cout<<"����"<<endl;
	std::cout<<"�������!"<<endl;

	//����궨���
	std::cout<<"------- ��ʼ����궨��������� --------"<<endl;
	Mat ratation_matrix=Mat(3, 3, CV_32FC1, Scalar::all(0) );  //����ÿ��ͼ�����ת����
	fout<<"������ڲ�������:"<<endl;
	fout<<cameraMatrix<<endl<<endl;
	fout<<"����Ļ���ϵ��:"<<endl;
	fout<<distCoeffs<<endl<<endl<<endl;
	for(int i=0; i<image_cout; i++)
	{
		fout<<"��"<<i+1<<"��ͼ�����ת������"<<endl;
		fout<<tvecsMat[i]<<endl;

		//����ת����ת��Ϊ��Ӧ����ת����:�޵����˹��Rodrigues���任
		//һ������������ת����ȼ���������ĳ�ַ�ʽ������ת����ת�����ĳ��ȣ�ģ����ʾ������ʱ����ת�ĽǶȣ����ȣ�
		Rodrigues(tvecsMat[i], ratation_matrix);
		fout<<"��"<<i+1<<"��ͼ�����ת����"<<endl;
		fout<<ratation_matrix<<endl;
		fout<<"��"<<i+1<<"��ͼ���ƽ��������"<<endl;
		fout<<rvecsMat[i]<<endl;
	}
	std::cout<<"��ɱ���"<<endl;
	fout<<endl;

	//��ʾ�궨���:������õ�����ڲκ���Σ���ͼ����л���Ľ���
	Mat mapx=Mat(image_size, CV_32FC1);
	Mat mapy=Mat(image_size, CV_32FC1);
	Mat R=Mat::eye(3, 3, CV_32F);  //Mat::eye()������һ�����ָ����С�����͵ľ���
	std::cout<<"�������ͼ��"<<endl;
	string imageFilename;
	std::stringstream StrStm;
	for(int i=0; i!=image_cout; i++)
	{
		std::cout<<"Frame #"<<i+1<<"������"<<endl;

		//��1��	��initUndistortRectifyMap��remap����������ϣ�initUndistortRectifyMap�������ӳ�䣬remap����õ�ӳ��Ӧ�õ�ͼ����
		/*
		initUndistortRectifyMap�������ӳ��
		��һ��������֮ǰ��õ�����ڲξ���
		�ڶ���������֮ǰ��õ�����������
		��������������ѡ������룬�ǵ�һ�͵ڶ��������֮�����ת����
		���ĸ�����������Ľ������3*3�������
		���������������ɼ�����ʧ���ͼ��ߴ�
		������������������ߺ͵ڰ˸���������������
		���߸��͵ڰ˸������������X/Y������ӳ�����
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
		remap����õ�ӳ��Ӧ�õ�ͼ����
		��һ����������������ԭʼͼ��
		�ڶ��������������������ͼ���������ͼ������ͬ�����ͺʹ�С
		�����͵��ĸ�������X/Y�����ӳ��
		���������������ͼ��Ĳ�ֵ��ʽ
		����������������߽���䷽ʽ
		*/
		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);

		//��2��undistort����ʵ��
		/*��һ����������������ԭʼͼ��
		�ڶ��������������������ͼ���������ͼ������ͬ�����ͺʹ�С
		������������֮ǰ��õ�����ڲξ���
		���ĸ�������֮ǰ��õ�����������*/
		//��1�ȷ�2Ч�ʸ���
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
	std::cout<<"������"<<endl;
	

	return 0;
}
 