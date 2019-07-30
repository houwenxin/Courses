#include "stdafx.h" 

#include "FigureManager.h"
#include "BlackBoard.h"
#include <vector>

//using std::cout;
//using std::endl;
using namespace std;
void FigureManager::input(std::istream &is)
{
    // 1����ӡ�˵������û�ѡ������ͼ��
	// �˵���ʽΪ
	// input type(1 : Circle, 2 : Line, 3 : Rectangle, 0 : Quit)

	// ���벻Ҫ��std::cin
	// �ñ��������������is���÷�ͬcin

	// 2���û���������(1/2/3/0)ѡ��󣬸��ݲ�ͬ��ͼ�Σ���ʾ����ͼ�εĲ���
	// ����Բ�Ĳ�������ΪԲ��x��y���뾶����˰�����ʾ����������������½��
	int language,input,_x,_y,_radius,_x1,_y1,_x2,_y2,_left,_right,_top,_bottom;
	input = 8848;
	language = 1;
	while(input != 0)
	{
		//��װţ�Ƶ���Ӣ˫���л�
		if(language == 1){
			cout << "Input type(1 : Circle, 2 : Line, 3 : Rectangle, 0 : Quit/Draw)" << endl;
			cout <<"You can change your language by inputing \"-1\" (The figures you've made will be kept.) ."<<endl;
		}
		else if(language == 2){
			cout << "����ͼ�����ͣ�1��Բ��2��ֱ�ߣ�3�����Σ�0���˳�/��ͼ��"<<endl;
			cout <<"��Ҳ��������-1ѡ�����ԡ���֮ǰ����ͼ�ᱣ����"<<endl;
		}
		is >> input;
		if(input == -1)
		{
			system("cls");
			cout << "Choosing Language"<<endl;
			cout << "1.English" <<endl;
			cout << "2.����" <<endl;
			is >> language;
			system("cls");
		}
		if(input == 1)
		{
			if(language == 1)
			{
				cout << "Center X: "<<endl; 
				is >> _x; 
				cout << "Center Y: "<<endl; 
				is >> _y;
				std::cout << "Radius: "<<endl; 
				is >> _radius; 
			}
			else if(language == 2)
			{
				cout << "Բ�ĺ�����: "<<endl; 
				is >> _x; 
				cout << "Բ��������: "<<endl; 
				is >> _y;
				std::cout << "�뾶: "<<endl; 
				is >> _radius; 
			}
			FlexibleFig circle = FlexibleFig(input, _x,_y,_radius);
			Figs.push_back(circle);
		}
		if(input == 2)
		{
			if(language == 1)
			{
				std::cout << "X1: "<<endl; 
				is >> _x1; 
				std::cout << "Y1: "<<endl; 
				is >> _y1; 
				std::cout << "X2: "<<endl; 
				is >> _x2; 
				std::cout << "Y2: "<<endl; 
				is >> _y2; 
			}
			else if(language == 2)
			{
				std::cout << "������һ: "<<endl; 
				is >> _x1; 
				std::cout << "������һ: "<<endl; 
				is >> _y1; 
				std::cout << "�������: "<<endl; 
				is >> _x2; 
				std::cout << "�������: "<<endl; 
				is >> _y2; 
			}
			FlexibleFig Line = FlexibleFig(input, _x1, _y1, _x2, _y2);
			Figs.push_back(Line);
		}
		if(input == 3)
		{
			if(language == 1)
			{
				std::cout << "Left: "<<endl; 
				is >> _left; 
				std::cout << "Top: "<<endl; 
				is >> _top; 	
				std::cout << "Right: "<<endl; 
				is >> _right; 
				std::cout << "Bottom: "<<endl; 
				is >> _bottom; 
			}
			else if(language == 2)
			{
				std::cout << "��ߵĺ�����: "<<endl; 
				is >> _left; 
				std::cout << "�ϱߵ�������: "<<endl; 
				is >> _top; 	
				std::cout << "�ұߵĺ�����: "<<endl; 
				is >> _right; 
				std::cout << "�±ߵ�������: "<<endl; 
				is >> _bottom; 
			}
			FlexibleFig rectangle = FlexibleFig(input, _left, _top, _right, _bottom);
			Figs.push_back(rectangle);
		}
		if(input == 0)
		{
			//system("cls");
			return;
		}
#if 0
	std::cout << "Center X: "; 
	is >> _x; 

	std::cout << "Center Y: "; 
	is >> _y; 

	std::cout << "Radius: "; 
	is >> _radius;  
#endif

	// �߲���Ϊ�˵�1 X��Y���꣬�˵�2 X���ꡢY���ꡣ
#if 0
	std::cout << "X1: "; 
	is >> _x1; 
	std::cout << "Y1: "; 
	is >> _y1; 

	std::cout << "X2: "; 
	is >> _x2; 
	std::cout << "Y2: "; 
	is >> _y2; 
#endif

	// ���β���Ϊ���϶���x��y�����¶���x��y
#if 0
	std::cout << "Left: "; 
	is >> _left; 
	std::cout << "Top: "; 
	is >> _top; 

	std::cout << "Right: "; 
	is >> _right; 
	std::cout << "Bottom: "; 
	is >> _bottom; 
#endif

	// 3������ò����󣬽�ͼ�α�������������֮�����

	// 4���ص�1��������ӡ�˵���ֱ���û�ѡ��0���˳�input����
	
	}

}

void FigureManager::display(BlackBoard &board)
{
	// ������input�������ͼ�����������λ���
	vector <FlexibleFig>::iterator iter = Figs.begin();
	for(iter = Figs.begin(); iter != Figs.end(); iter++)
	{
		board.DrawCircle(iter->x,iter->y,iter->radius);//Circle
		board.DrawLine(iter->x1,iter->y1,iter->x2,iter->y2);//Line
		board.DrawLine(iter->left,iter->top,iter->left,iter->bottom);//Rectangle1
		board.DrawLine(iter->right,iter->top,iter->right,iter->bottom);//Rectangle2
		board.DrawLine(iter->left,iter->top,iter->right,iter->top);//Rectangle3
		board.DrawLine(iter->left,iter->bottom,iter->right,iter->bottom);//Rectangle4
	}
#if 0
    vector <Circle>::iterator iter_Circle = Circles.begin();
	vector <Line>::iterator iter_Line = Lines.begin();
	vector <Rectangle>::iterator iter_Rectangle = Rectangles.begin();
	for(iter_Circle=Circles.begin();iter_Circle!=Circles.end();iter_Circle++)
		{
			board.DrawCircle(iter_Circle->x,iter_Circle->y,iter_Circle->radius);
		}
	for(iter_Line=Lines.begin();iter_Line!=Lines.end();iter_Line++)
		{
			board.DrawLine(iter_Line->x1,iter_Line->y1,iter_Line->x2,iter_Line->y2);
		}
	for(iter_Rectangle = Rectangles.begin();iter_Rectangle != Rectangles.end();iter_Rectangle++)
	{
		board.DrawLine(iter_Rectangle->left,iter_Rectangle->top,iter_Rectangle->left,iter_Rectangle->bottom);
		board.DrawLine(iter_Rectangle->right,iter_Rectangle->top,iter_Rectangle->right,iter_Rectangle->bottom);
		board.DrawLine(iter_Rectangle->left,iter_Rectangle->top,iter_Rectangle->right,iter_Rectangle->top);
		board.DrawLine(iter_Rectangle->left,iter_Rectangle->bottom,iter_Rectangle->right,iter_Rectangle->bottom);
	}
#endif
	// ����board�ṩ��DrawCircle��DrawLine������ͼ����������һ��Բ����30,30λ�ã��뾶Ϊ100��Բ
	//board.DrawCircle(30, 30, 100); 

	// �پ����������Σ��þ������Ͻ�Ϊ(30, 20)�����½�Ϊ��200,300��
	//board.DrawLine(30, 20, 30, 300); 
	//board.DrawLine(30, 300, 200, 300); 
	//board.DrawLine(200, 300, 200, 20); 
	//board.DrawLine(200, 20, 30, 20); 
}

// ������ʵ����ҪһЩ��Ҫ�ĳ�ʼ�����ɷ�����������У�main���������
// ���û�У������
void InitiateFigureManager()
{

}


