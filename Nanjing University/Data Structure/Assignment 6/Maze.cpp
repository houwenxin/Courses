// Maze.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Maze.h"

int Maze[4][4] = {
	{0, 0, 0, 1},
	{0, 1, 0, 0},
	{0, 0, 0, 1},
	{1, 0, 0, 0}
};
int main()
{
	BFSearch(0, 0, Maze);

    return 0;
}

