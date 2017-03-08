/*
------------------------------------------------
Author: CIEL
Date: 2017/01/16
Function: 
1.几种颜色空间的图片可视化
2.相机标定
3.阴影检测
(1)色度差阴影检测
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
#define WIDTH 1408  //HoloLens图像宽度
#define HEIGHT 792  //HoloLens图像高度

int resultRGB_B[WIDTH][HEIGHT],resultRGB_G[WIDTH][HEIGHT],resultRGB_R[WIDTH][HEIGHT];  //保存阴影检测结果的RGB分量
*/

int main()
{
	/*
	//各种颜色空间及相应通道的可视化
 	visualization_RGB();   //RGB
	visualization_HSI();   //HSI
	visualization_LAB();   //LAB
	visualization_HSV();   //HSV
	visualization_YCbCr();  //YCbCr
	visualization_C1C2C3();  //C1C2C3 
	visualization_L1L2L3();  //L1L2L3
	*/

	//相机标定
//	camera_calibrate();  

	//阴影检测
	shadowDetection();
	

	system("pause");
	return 0;
} 