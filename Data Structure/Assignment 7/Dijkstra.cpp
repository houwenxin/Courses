// Dijkstra.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Dijkstra.h"

int main()
{
	ALGraph G;
	CreateUDN(G);
	InitVisited(G);
	Dijkstra(G,0);
#if 1
	for (int i = 0; i < G.vexnum; i++) {
		printf("%d ", parent[i]);
	}
#endif
	DestroyGraph(G);
    return 0;
}

