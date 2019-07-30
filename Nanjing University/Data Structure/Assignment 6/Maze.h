#pragma once
#ifndef MAZE_H_
#define MAZE_H_

#include <stdio.h>

#define OK 1
typedef int Status;

int Move_X[4] = { 0 ,0, 1, -1 };
int Move_Y[4] = { 1, -1, 0, 0 };

struct node {
	int x;
	int y;
	int pre;
}path[100];

void TraceBack(int i) {
	if (path[i].pre != -1)
		TraceBack(path[i].pre);
	printf("(%d, %d) ", path[i].x, path[i].y);
}
Status BFSearch(int x_start, int y_start, int Maze[4][4]) {
	int front = -1, rear = -1;
	path[++rear].x = x_start;
	path[rear].y = y_start;
	path[rear].pre = -1;
	while (front != rear) {
		front++;
		for (int i = 0; i < 4; i++) {
			int pathX = path[front].x + Move_X[i];
			int pathY = path[front].y + Move_Y[i];
			if (pathX < 0 || pathX > 3 || pathY < 0 || pathY > 3 || Maze[pathX][pathY]) {
				continue;
			}	
			else {
				Maze[pathX][pathY] = 1;
				path[++rear].x = pathX;
				path[rear].y = pathY;
				path[rear].pre = front;
			}
			if (pathX == 3 && pathY == 3) {
				TraceBack(rear);
				break;
			}
		}
	}
	return OK;
}
#endif