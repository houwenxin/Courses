// Draw.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include <iostream>
#include <fstream>

#include "BlackBoard.h"
#include "FigureManager.h"

// �ṩ��ͼ�����Ķ���
BlackBoard board; 

// �ɺ���
void ReshapeCallback(int width, int height)
{
	board.UpdateWindowSize(width, height); 
}

// �������ڴ����������Ļص�������ڣ�����ֻ������һ���£���q�˳����򣬿ɺ���
// ע�⣬����ļ��������Ǵ��ڵļ������룬�����������е�
void KeyboardCallback(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
		exit(0);
		break; 
	}

}

// �������ڻ��ƵĻص�������ڣ����ջ����FigureManager��display����
// ������ҵ����Ķ�DisplayCallback������ͼ��Ӧ��FigureManager����ɡ�
void DisplayCallback()
{
	board.Clear(); 

	FigureManager::handle().display(board); 

	board.Flush(); 
}

int _tmain(int argc, _TCHAR* argv[])
{
	// ������ʵ����ҪһЩ��Ҫ�ĳ�ʼ�����ɷ�����������У�main���������
	// ���û�У������
	InitiateFigureManager(); 

	// ������л����뷽ʽ��
	// 1����test.txt�ļ����룬�Է������ʱ���������ʱ��
	// 2�������������룬�԰�����û����ȫ��д�ô���ʱ���޷���test.txt����

#if 1
	std::ifstream in("test.txt");  
	if (! in.is_open())  
	{ 
		std::cout << "Error opening file"; 
		exit (1); 
	}  
	FigureManager::handle().input(in); 
#else
	FigureManager::handle().input(std::cin); 
#endif

	// ���´������ڳ�ʼ�����ڵȡ��ɺ���
	board.InitCommandLine(&argc, (char **)argv); 
	board.InitWindowSize(1000, 800); 

	board.InitDisplayCallback(DisplayCallback); 
	board.InitKeyboardCallback(KeyboardCallback); 
	board.InitReshapeCallback(ReshapeCallback); 

	board.Show(); 

	return 0;
}

