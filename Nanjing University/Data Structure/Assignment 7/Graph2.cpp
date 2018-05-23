// Graph2.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Graph2.h"

int main()
{
// Basic
#if 1
	ALGraph G;
	CreateDG(G);
	InitVisited(G);
	for (int i = 0; i < G.vexnum; i++)
		if (!DFN[i])  tarjan(G, i); 
	DestroyGraph(G);
	system("PAUSE");
#endif
// Advance
#if 0
	ALGraph G;
	CreateUDN(G);
	DestroyGraph(G);
#endif
    return 0;
}

