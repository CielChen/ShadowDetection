#include <iostream>
#include <highgui/highgui.hpp>

//C1C2C3空间图片可视化
int visualization_C1C2C3();

//将C1C2C3空间的三个分量组合起来，便于显示
IplImage* catC1C2C3Image(CvMat* C1C2C3_C1, CvMat* C1C2C3_C2, CvMat* C1C2C3_C3);

//C1C2C3各分量可视化
int visualization_C1_C2_C3();

//求最大值
int max_ab(int a, int b);

