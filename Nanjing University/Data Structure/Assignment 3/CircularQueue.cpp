// CircularQueue.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include "CircularQueue.h"

int main()
{	
	clock_t start, finish;
	double TotalTime;
	start = clock();
#if 0 //Basic Part
	CirQueue Q;
	InitQueue(Q);
	for (int i = 0; i < 100; i++) {
		AddElem(Q, i);
	}
	for (int i = 0; i < 100; i++) {
		int temp = DeleteElem(Q);
		printf("%d Deleted !\n", temp);
	}
	DeleteQueue(Q);
#endif
#if 1 //Advanced Part
	int n = 5;
	int k = 3;
	int a[] = { 2,3,1,4,5 };
	CirQueue B;
	CirQueue Q;
	InitQueue(Q); InitQueue(B);
	for (int i = 0; i < n; i++) {
		if (i >= k) {
			AddElem(B, Q.Data[Q.front]);
			if (a[i - k] == Q.Data[Q.front]) {
				DeleteElem(Q);
			}
		}
		while ((Q.rear != Q.front) && Q.Data[(Q.rear - 1 + MAXSIZE) % MAXSIZE] > a[i]) { //比要进来的那个小的全部送出去
			DeleteRearElem(Q);
		}
		
		AddElem(Q, a[i]);
		//PrintQueue(Q);
		printf("Division------\n");
	}
	AddElem(B, Q.Data[Q.front]);
	PrintQueue(B);
	DeleteQueue(Q);
	DeleteQueue(B);
#endif

	finish = clock();
	TotalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("Execution time: %fs\n", TotalTime);
    return 0;
}

