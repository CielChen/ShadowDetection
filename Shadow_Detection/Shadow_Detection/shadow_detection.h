#include <iostream>
#include <highgui/highgui.hpp>


//��������2����
double norm2(int b,int g,int r);

//ɫ�Ȳ���Ӱ���
int chromaticityDiffer();

//���Ȳ���Ӱ���
int brightnessDiffer();

//�ֲ����ȱ�
int localRelation();

//������ͨ��İ�Χ��ϵ�Ż���Ӱ������
int spatialAjustment();
//�Զ������ֵ����
void on_ThreshChange(int, void*);
//������һ���ֶ���ֵЧ��������С��ͨ���ͼ������Ż�
void improvedSpace();

//��Ӱ����㷨
int shadowDetection();