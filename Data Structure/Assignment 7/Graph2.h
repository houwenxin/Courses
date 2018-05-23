#pragma warning(disable:4996) 
#pragma once
#ifndef GRAPH2_H_
#define GRAPH2_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VERTEX_NUM 100
#define OK 1
typedef double InfoType;
typedef int VertexType;
typedef int Status;
typedef enum {
	DG, DN, UDG, UDN
}GraphKind;

//For tarjan algorithm
int DFN[1001] = { 0 }, LOW[1001] = { 0 };
int stack[1001], cnt, index, tot = 0;


typedef struct EdgeNode {
	int adjvex;
	struct EdgeNode *nextarc;
	InfoType weight;
}EdgeNode;
typedef struct VNode {
	VertexType coordinate[2];
	EdgeNode *firstarc;
}VNode, AdjList[MAX_VERTEX_NUM];
typedef struct {
	AdjList vertices;
	int vexnum, edgenum;
	int kind;
	int visited[MAX_VERTEX_NUM] = { 0 };
}ALGraph;

Status CreateDG(ALGraph &G) {
	printf("Please enter vertex number: ");
	scanf("%d", &G.vexnum);
	printf("\nPlease enter edge number: ");
	scanf("%d", &G.edgenum);
	printf("\nPlease enter the vertexs' information: ");
	for (int i = 0; i < G.vexnum; i++) {
		scanf("%d", &G.vertices[i].coordinate[0]);
		G.vertices[i].firstarc = NULL;
	}
	printf("\nPlease enter the edges' information , input format: start,end\n");
	EdgeNode *new_edge1, *new_edge2;
	int j, k;
	for (int i = 0; i < G.edgenum; i++) {
		scanf("%d,%d", &j, &k);
		j = j - 1;
		k = k - 1;
		new_edge1 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge1->adjvex = j;
		new_edge1->nextarc = G.vertices[k].firstarc;
		G.vertices[k].firstarc = new_edge1;
	}
	return OK;
}

Status CreateUDG(ALGraph &G) {
	printf("Please enter vertex number: ");
	scanf("%d", &G.vexnum);
	printf("\nPlease enter edge number: ");
	scanf("%d", &G.edgenum);
	printf("\nPlease enter the vertexs' information: ");
	for (int i = 0; i < G.vexnum; i++) {
		scanf("%d", &G.vertices[i].coordinate[0]);
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
		G.vertices[k].firstarc = new_edge1;
#if 1
		new_edge2 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge2->adjvex = k;
		new_edge2->nextarc = G.vertices[j].firstarc;
		G.vertices[j].firstarc = new_edge2;
#endif
	}
	return OK;
}

InfoType Cost(VNode P1, VNode P2) {
	InfoType cost = sqrt(pow(P1.coordinate[0] - P2.coordinate[0], 2) + pow(P1.coordinate[1] - P2.coordinate[1], 2));
	return cost;
}
Status CreateUDN(ALGraph &G) {
	printf("Please enter vertex number: ");
	scanf("%d", &G.vexnum);
	printf("\nPlease enter edge number: ");
	scanf("%d", &G.edgenum);
	printf("\nPlease enter the vertexs' information, format: x,y\n");
	for (int i = 0; i < G.vexnum; i++) {
		scanf("%d,%d", &G.vertices[i].coordinate[0],&G.vertices[i].coordinate[1]);
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
		InfoType cost = Cost(G.vertices[j], G.vertices[k]);
		new_edge1->weight = cost;
		G.vertices[k].firstarc = new_edge1;
#if 1
		new_edge2 = (EdgeNode *)malloc(sizeof(EdgeNode));
		new_edge2->adjvex = k;
		new_edge2->nextarc = G.vertices[j].firstarc;
		new_edge2->weight = cost;
		G.vertices[j].firstarc = new_edge2;
#endif
	}
	return OK;
}

struct {	//Used for PRIM.
	VertexType adjvex;
	double lowcost;
} closedge[MAX_VERTEX_NUM];

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
	printf("%d ", G.vertices[v].coordinate[0]);
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

	printf("%d ", G.vertices[v].coordinate[0]);
	G.visited[v] = 1;
	Queue[++rear] = v;
	while (front != rear) {	//When Queue is not empty.
		v = Queue[++front];
		EdgeNode *p = G.vertices[v].firstarc;
		while (p != NULL) {
			if (G.visited[p->adjvex] == 0) {
				printf("%d ", G.vertices[p->adjvex].coordinate[0]);
				G.visited[p->adjvex] = 1;
				Queue[++rear] = p->adjvex;
			}	
			p = p->nextarc;
		}
	}
	printf("\n");
	return OK;
}

int min(int a, int b) {
	return a < b ? a : b;
}

Status tarjan(ALGraph &G, int v) {
	DFN[v] = LOW[v] = ++tot;
	stack[++index] = v;
	G.visited[v] = 1; // Equal to Instack[v] = 1
	EdgeNode *p = G.vertices[v].firstarc;
	while (p != NULL) {
		if (!DFN[p->adjvex]) {
			tarjan(G, p->adjvex);
			LOW[v] = min(LOW[v], LOW[p->adjvex]);
		}
		else if (G.visited[p->adjvex] == 1) {
			LOW[v] = min(LOW[v], DFN[p->adjvex]);
		}
		p = p->nextarc;
	}
	if (LOW[v] == DFN[v]) {
		do {
			printf("%d ", G.vertices[stack[index]].coordinate[0]);
			G.visited[stack[index]] = 0;
			index--;
		} while (v != stack[index + 1]);
		printf("\n");
	}
	return OK;
}
#endif