#include <iostream>
#include <highgui/highgui.hpp>

//HSI�ռ�ͼƬ���ӻ�
int visualization_HSI();

//��HSI�ռ�������������������������ʾ
IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I);

//HSI���������ӻ�
int visualization_H_S_I();

//����Сֵ
int min(int a, int b, int c);