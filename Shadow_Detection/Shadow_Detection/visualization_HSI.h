#include <iostream>
#include <highgui/highgui.hpp>

//HSI空间图片可视化
int visualization_HSI();

//将HSI空间的三个分量组合起来，便于显示
IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I);

//HSI各分量可视化
int visualization_H_S_I();

//求最小值
int min(int a, int b, int c);