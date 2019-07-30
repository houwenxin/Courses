#pragma warning(disable:4996) 
#pragma once
#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTEX_NUM 100
#define OK 1
typedef int InfoType;
typedef int VertexType;
typedef int Status;

typedef struct EdgeNode {
	int adjvex;
	struct EdgeNode *nextarc;
	InfoType *info;
}EdgeNode;
typedef struct VNode {
	VertexType data;
	EdgeNode *firstarc;
}VNode, AdjList[MAX_VERTEX_NUM];
typedef struct {
	AdjList vertices;
	int vexnum, edgenum;
	int kind;
	int visited[MAX_VERTEX_NUM] = { 0 };
}ALGraph;

Status CreateGraph(ALGraph &G) {
	printf("Please enter vertex number: ");
	scanf("%d", &G.vexnum);
	printf("\nPlease enter edge number: ");
	scanf("%d", &G.edgenum);
	printf("\nPlease enter the vertexs' information: ");
	for (int i = 0; i < G.vexnum; i++) {
		scanf("%d", &G.vertices[i].data);
		G.vertices[i].firstarc = NULL;
	}
	printf("\nPlease enter the edges' information , input format: i,j\n");
	EdgeNode *new_edge1, *new_edge2;
	int j, k;
	for (int i = 0; i < G.edgenum; i++) {
		scanf("%d,%d", &j, &k);
		new_edge1 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge1->adjvex = k;
		new_edge1->nextarc = G.vertices[j].firstarc;
		G.vertices[j].firstarc = new_edge1;

		new_edge2 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge2->adjvex = j;
		new_edge2->nextarc = G.vertices[k].firstarc;
		G.vertices[k].firstarc = new_edge2;
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
Status DFSTraverse(ALGraph &G, int v) {
	printf("%d ", G.vertices[v].data);
	G.visited[v] = 1;
	EdgeNode *p = G.vertices[v].firstarc;
	while (p != NULL) {
		if (G.visited[p->adjvex] == 0)
			DFSTraverse(G, p->adjvex);
		p = p->nextarc;
	}
	return OK;
}

Status BFSTraverse(ALGraph &G, int v) {
	int Queue[MAX_VERTEX_NUM];
	int front = -1, rear = -1; //Initialize Queue.

	printf("%d ", G.vertices[v].data);
	G.visited[v] = 1;
	Queue[++rear] = v;
	while (front != rear) {	//When Queue is not empty.
		v = Queue[++front];
		EdgeNode *p = G.vertices[v].firstarc;
		while (p != NULL) {
			if (G.visited[p->adjvex] == 0) {
				printf("%d ", G.vertices[p->adjvex].data);
				G.visited[p->adjvex] = 1;
				Queue[++rear] = p->adjvex;
			}	
			p = p->nextarc;
		}
	}
	printf("\n");
	return OK;
}
#endif