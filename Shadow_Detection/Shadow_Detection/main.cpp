/*
------------------------------------------------
Author: CIEL
Date: 2017/01/16
Function: 
1.������ɫ�ռ��ͼƬ���ӻ�
2.����궨
3.��Ӱ���
(1)ɫ�Ȳ���Ӱ���
------------------------------------------------
*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <core/affine.hpp>
#include <highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "visualization_RGB.h"
#include "visualization_HSI.h"
#include "visualization_LAB.h"
#include "visualization_HSV.h"
#include "visualization_YCbCr.h"
#include "visualization_C1C2C3.h"
#include "visualization_L1L2L3.h"
#include "camera_calibrate.h"
#include "shadow_detection.h"

using namespace cv;
using namespace std; 

/*
#define WIDTH 1408  //HoloLensͼ����
#define HEIGHT 792  //HoloLensͼ��߶�

int resultRGB_B[WIDTH][HEIGHT],resultRGB_G[WIDTH][HEIGHT],resultRGB_R[WIDTH][HEIGHT];  //������Ӱ�������RGB����
*/

int main()
{
	/*
	//������ɫ�ռ估��Ӧͨ���Ŀ��ӻ�
 	visualization_RGB();   //RGB
	visualization_HSI();   //HSI
	visualization_LAB();   //LAB
	visualization_HSV();   //HSV
	visualization_YCbCr();  //YCbCr
	visualization_C1C2C3();  //C1C2C3 
	visualization_L1L2L3();  //L1L2L3
	*/

	//����궨
//	camera_calibrate();  

	//��Ӱ���
	shadowDetection();
	

	system("pause");
	return 0;
} 