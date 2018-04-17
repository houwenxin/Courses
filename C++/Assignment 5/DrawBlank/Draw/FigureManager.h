#ifndef _FIGUREMANAGER_H_
#define _FIGUREMANAGER_H_

#include <iostream>
#include <vector>
#include "BlackBoard.h"
using namespace std;

class BlackBoard; 

class Circle
{
public:
	int x;
	int y;
	int radius;
	Circle():x(0),y(0),radius(0){}
};
class Line
{
public:
	int x1;
	int x2;
	int y1;
	int y2;
	Line():x1(0),x2(0),y1(0),y2(0){}
};
class Rectangle
{
public:
	int left;
	int right;
	int top;
	int bottom;
	Rectangle():left(0),right(0),top(0),bottom(0){}
};
class FlexibleFig:public Circle, public Line, public Rectangle //为了满足扩展性的要求，定义一个继承类
{
public:
	FlexibleFig(int mode = 0 , int P1=0, int P2=0, int P3 = 0, int P4=0)
	{
		if(mode == 1)
		{
			x = P1;
			y = P2;
			radius = P3;
		}
		if(mode == 2)
		{
			x1 = P1;
			y1 = P2;
			x2 = P3;
			y2 = P4;
		}
		else if(mode == 3)
		{
			left = P1;
			top = P2;
			right = P3;
			bottom = P4;
		}
	}
};


class FigureManager
{
public:
	static FigureManager &handle()
	{
		static FigureManager manager; 
		return manager; 
	}
    
    // FigureManager类析构函数
    virtual ~FigureManager() { }
 
    // FigureManager类接口.
public:
	void input(std::istream &is); 
    void display(BlackBoard &board);
private:
	vector <FlexibleFig> Figs;
}; // class FigureManager类定义结束.

void InitiateFigureManager(); 

#endif // #ifndef _FIGUREMANAGER_H_
