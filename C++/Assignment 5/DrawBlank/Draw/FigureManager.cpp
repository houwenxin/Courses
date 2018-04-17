#include "stdafx.h" 

#include "FigureManager.h"
#include "BlackBoard.h"
#include <vector>

//using std::cout;
//using std::endl;
using namespace std;
void FigureManager::input(std::istream &is)
{
    // 1、打印菜单，让用户选择输入图形
	// 菜单格式为
	// input type(1 : Circle, 2 : Line, 3 : Rectangle, 0 : Quit)

	// 输入不要用std::cin
	// 用本函数的输入参数is，用法同cin

	// 2、用户输入数字(1/2/3/0)选择后，根据不同的图形，提示输入图形的参数
	// 其中圆的参数依次为圆心x、y、半径，因此包含提示的输入代码类似以下结果
	int language,input,_x,_y,_radius,_x1,_y1,_x2,_y2,_left,_right,_top,_bottom;
	input = 8848;
	language = 1;
	while(input != 0)
	{
		//假装牛逼的中英双语切换
		if(language == 1){
			cout << "Input type(1 : Circle, 2 : Line, 3 : Rectangle, 0 : Quit/Draw)" << endl;
			cout <<"You can change your language by inputing \"-1\" (The figures you've made will be kept.) ."<<endl;
		}
		else if(language == 2){
			cout << "输入图形类型（1：圆，2：直线，3：矩形，0：退出/作图）"<<endl;
			cout <<"您也可以输入-1选择语言。（之前做的图会保留）"<<endl;
		}
		is >> input;
		if(input == -1)
		{
			system("cls");
			cout << "Choosing Language"<<endl;
			cout << "1.English" <<endl;
			cout << "2.中文" <<endl;
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
				cout << "圆心横坐标: "<<endl; 
				is >> _x; 
				cout << "圆心纵坐标: "<<endl; 
				is >> _y;
				std::cout << "半径: "<<endl; 
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
				std::cout << "横坐标一: "<<endl; 
				is >> _x1; 
				std::cout << "纵坐标一: "<<endl; 
				is >> _y1; 
				std::cout << "横坐标二: "<<endl; 
				is >> _x2; 
				std::cout << "纵坐标二: "<<endl; 
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
				std::cout << "左边的横坐标: "<<endl; 
				is >> _left; 
				std::cout << "上边的纵坐标: "<<endl; 
				is >> _top; 	
				std::cout << "右边的横坐标: "<<endl; 
				is >> _right; 
				std::cout << "下边的纵坐标: "<<endl; 
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

	// 线参数为端点1 X、Y坐标，端点2 X坐标、Y坐标。
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

	// 矩形参数为左上顶点x、y，右下顶点x、y
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

	// 3，输入好参数后，将图形保存下来，用于之后绘制

	// 4，回到1，继续打印菜单，直到用户选择0，退出input函数
	
	}

}

void FigureManager::display(BlackBoard &board)
{
	// 将所有input中输入的图形在这里依次画出
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
	// 借助board提供的DrawCircle和DrawLine函数画图，举例，画一个圆心在30,30位置，半径为100的圆
	//board.DrawCircle(30, 30, 100); 

	// 再举例，画矩形，该矩形左上角为(30, 20)，右下角为（200,300）
	//board.DrawLine(30, 20, 30, 300); 
	//board.DrawLine(30, 300, 200, 300); 
	//board.DrawLine(200, 300, 200, 20); 
	//board.DrawLine(200, 20, 30, 20); 
}

// 如果你的实现需要一些必要的初始化，可放在这个函数中，main函数会调用
// 如果没有，则忽略
void InitiateFigureManager()
{

}


