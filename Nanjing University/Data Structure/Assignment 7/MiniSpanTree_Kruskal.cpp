// MiniSpanTree_Kruskal.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "MiniSpanTree_Kruskal.h"



int main()
{
	MGraph G;
	CreateUDN(G);
	MiniSpanTree_Kruskal(G);
	return 0;
}


