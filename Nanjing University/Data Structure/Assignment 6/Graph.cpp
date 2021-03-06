// Graph.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Graph.h"

int main()
{
	ALGraph G;
	CreateGraph(G);
	printf("DFS:\n");
	InitVisited(G);
	DFSTraverse(G, 0);
	printf("\nBFS:\n");
	InitVisited(G);
	BFSTraverse(G, 0);
	DestroyGraph(G);
    return 0;
}

