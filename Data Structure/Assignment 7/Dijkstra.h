#pragma warning(disable:4996) 
#pragma once
#ifndef DIJKSTRA_H_
#define DIJKSTRA_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VERTEX_NUM 100
#define OK 1
typedef int VertexType;
typedef int Status;
typedef double InfoType;

typedef struct EdgeNode {
	int adjvex;
	struct EdgeNode *nextarc;
	InfoType weight;
}EdgeNode;
typedef struct VNode {
	VertexType data[2];
	EdgeNode *firstarc;
}VNode, AdjList[MAX_VERTEX_NUM];
typedef struct {
	AdjList vertices;
	int vexnum, edgenum;
	int kind;
	int visited[MAX_VERTEX_NUM] = { 0 };
}ALGraph;

Status CreateUDN(ALGraph &G) {
	printf("Please enter vertex number: ");
	scanf("%d", &G.vexnum);
	printf("\nPlease enter edge number: ");
	scanf("%d", &G.edgenum);
	printf("\nPlease enter the vertexs' information, format: x,y\n");
	for (int i = 0; i < G.vexnum; i++) {
		scanf("%d,%d", &G.vertices[i].data[0], &G.vertices[i].data[1]);
		G.vertices[i].firstarc = NULL;
	}
	printf("\nPlease enter the edges' information , input format: i,j\n");
	EdgeNode *new_edge1, *new_edge2;
	int j, k;
	for (int i = 0; i < G.edgenum; i++) {
		scanf("%d,%d", &j, &k);
		new_edge1 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge1->adjvex = j;
		new_edge1->nextarc = G.vertices[k].firstarc;
		//InfoType cost = Cost(G.vertices[j], G.vertices[k]);
		//new_edge1->info = cost;
		G.vertices[k].firstarc = new_edge1;
#if 1
		new_edge2 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge2->adjvex = k;
		new_edge2->nextarc = G.vertices[j].firstarc;
		//new_edge2->info = cost;
		G.vertices[j].firstarc = new_edge2;
#endif
	}

	return OK;
}
Status DestroyGraph(ALGraph &G) {
	for (int i = 0; i < G.vexnum; i++) {
		while (G.vertices[i].firstarc != NULL) {
			EdgeNode *edge = G.vertices[i].firstarc;
			G.vertices[i].firstarc = G.vertices[i].firstarc->nextarc;
			free(edge);
			edge = NULL;
		}
	}
	return OK;
}
void InitVisited(ALGraph &G) {
	for (int i = 0; i < G.vexnum; i++) {
		G.visited[i] = 0;
	}
}

InfoType Cost(VNode P1, VNode P2) {
	InfoType cost = sqrt(pow(P1.data[0] - P2.data[0], 2) + pow(P1.data[1] - P2.data[1], 2));
	return cost;
}
struct DistanceNode {
	int ID;
	InfoType Weight;
} Distances[MAX_VERTEX_NUM];
typedef struct {
	DistanceNode *Distances;
	int front, rear;
}DistanceQueue;
void InitQ(DistanceQueue &Q) {
	Q.Distances = (DistanceNode *)malloc(sizeof(DistanceNode) * 100);
	Q.front = Q.rear = 0;
}
void PushQ(DistanceQueue &Q, DistanceNode node) {
	Q.Distances[Q.rear++] = node;
}
Status is_empty(DistanceQueue Q) {
	if (Q.front == Q.rear) return OK;
}
DistanceNode MinPopQ(DistanceQueue &Q) {
	int MinID;
	InfoType MinWeight = 0xFFFF;
	for (int i = Q.front; i < Q.rear; i++) {
		if (Q.Distances[i].Weight < MinWeight) {
			MinWeight = Q.Distances[i].Weight;
			MinID = Q.Distances[i].ID;
		}
	}
	DistanceNode temp = Q.Distances[MinID];
	for (int i = MinID; i > Q.front; i--) {
		Q.Distances[i] = Q.Distances[i - 1];
	}
	Q.front++;
	return temp;
}
DistanceQueue Q;
int parent[MAX_VERTEX_NUM];
void Dijkstra(ALGraph &G, int start) {
	for (int i = 0; i < G.vexnum; i++) {
		Distances[i].ID = i;
		Distances[i].Weight = 0xFFFF;
		G.visited[i] = 0;
		parent[i] = -1;
	}
	InitQ(Q);
	Distances[start].Weight = 0;
	PushQ(Q, Distances[start]);
	while (!is_empty(Q)) {
		DistanceNode cd = MinPopQ(Q);
		int u = cd.ID;
		if (G.visited[u] == 1) continue;
		G.visited[u] = 1;
		EdgeNode *p = G.vertices[u].firstarc;
		while (p != NULL) {
			int v = p->adjvex;
			if (!G.visited[v] && Distances[v].Weight > Distances[u].Weight + p->weight) {
				Distances[v].Weight = Distances[u].Weight + p->weight;
				parent[v] = u;
				PushQ(Q, Distances[v]);
			}
			p = p->nextarc;
		}
	}
}
#endif