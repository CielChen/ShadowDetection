#include <iostream>
#include <highgui/highgui.hpp>


//求向量的2范数
double norm2(int b,int g,int r);

//色度差阴影检测
int chromaticityDiffer();

//亮度差阴影检测
int brightnessDiffer();

//局部亮度比
int localRelation();

//利用连通域的包围关系优化阴影和物体
int spatialAjustment();
//自定义的阈值函数
void on_ThreshChange(int, void*);
//基于上一步手动阈值效果，用最小连通域对图像进行优化
void improvedSpace();

//阴影检测算法
int shadowDetection();