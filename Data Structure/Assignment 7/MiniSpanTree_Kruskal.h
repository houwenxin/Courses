#pragma warning(disable:4996)
#pragma once
#ifndef MINISPANTREE_KRUSKAL_H_
#define MINISPANTREE_KRUSKAL_H_

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VERTEX_NUM 100

double min, mincost = 0;
int parent[MAX_VERTEX_NUM] = { 0 };

typedef struct ArcCell {
	double weight;
}ArcCell, AdjMatrix[MAX_VERTEX_NUM][MAX_VERTEX_NUM];
typedef struct {
	int ID;
	int x, y;
} VertexType;
typedef struct {
	VertexType vexs[MAX_VERTEX_NUM];
	int vexnum, edgenum;
	AdjMatrix edges;
}MGraph;

double Cost(VertexType P1, VertexType P2) {
	double cost = sqrt(pow(P1.x - P2.x, 2) + pow(P1.y - P2.y, 2));
	return cost;
}
void CreateUDN(MGraph &G) {
	printf("Vexnum: ");
	scanf("%d", &G.vexnum);
	printf("Edgenum: ");
	scanf("%d", &G.edgenum);
	printf("Vex Information: x, y\n");
	for (int i = 1; i <= G.vexnum; i++) {
		scanf("%d,%d", &G.vexs[i].x, &G.vexs[i].y);
		G.vexs[i].ID = i;
	}
	for (int i = 1; i <= G.vexnum; i++) {
		for (int j = 1; j <= G.vexnum; j++) {
			G.edges[i][j].weight = 0xFFFFFFFF;
		}
	}
	int TotalEdges = G.vexnum * (G.vexnum - 1) / 2;
	for (int i = 1; i <= TotalEdges; i++){
		for (int j = 1; j <= TotalEdges; j++) {
			G.edges[i][j].weight = Cost(G.vexs[i], G.vexs[j]);
			G.edges[j][i] = G.edges[i][j];
		}
	}
	printf("Edge Information: x, y\n");
	int start, end; 
	for (int k = 0; k < G.edgenum; k++) {
		scanf("%d,%d", &start, &end);
		parent[end] = start;
		mincost += G.edges[start][end].weight;
	}
}


int find(int i)
{
	while (parent[i])
		i = parent[i];
	return i;
}
int uni(int i, int j)
{
	if (i != j)
	{
		parent[j] = i;
		return 1;
	}
	return 0;
}
void MiniSpanTree_Kruskal(MGraph G) {
	int i, j, k, a, b, u, v, ne = 1;
	printf("The edges of Minimum Cost Spanning Tree are\n");
	while (ne < G.vexnum-G.edgenum)
	{
		for (i = 1, min = 0xFFFFFFFF; i <= G.vexnum; i++)
		{
			for (j = 1; j <= G.vexnum; j++)
			{
				if (G.edges[i][j].weight < min)
				{
					min = G.edges[i][j].weight;
					a = u = i;
					b = v = j;
				}
			}
		}
		u = find(u);
		v = find(v);
		//printf("a,b,u,v: %d,%d,%d,%d\n", a,b,u, v);
		if (uni(u, v))
		{
			printf("%d edge (%d,%d) =%lf\n", ne++, a, b, min);
			mincost += min;
		}
		G.edges[a][b].weight = G.edges[b][a].weight = 0xFFFFFFFF;
	}
	printf("Minimum cost = %lf\n", mincost);
}

#endif