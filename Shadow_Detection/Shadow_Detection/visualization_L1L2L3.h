#include <iostream>
#include <highgui/highgui.hpp>

//L1L2L3空间图片可视化
int visualization_L1L2L3();

//将L1L2L3空间的三个分量组合起来，便于显示
IplImage* catL1L2L3Image(CvMat* L1L2L3_L1, CvMat* L1L2L3_L2, CvMat* L1L2L3_L3);

//L1L2L3各分量可视化
int visualization_L1_L2_L3();

