// Sort.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Sort.h"

int main()
{
	//Bubble Sort
	printf("Bubble Sort:\n");
	int test1[9] = { 3,21,4,1,3,4,5,6,12 };
	BubbleSort(test1, 9);
	for (int i = 0; i < 9; i++) {
		printf("%d ", test1[i]);
	}

	//Quick Sort
	printf("\nQuick Sort:\n");
	int test2[9] = { 3,21,4,1,3,4,5,6,12 };
	QuickSort(test2, 0, 8);
	for (int i = 0; i < 9; i++) {
		printf("%d ", test2[i]);
	}
	printf("\n");
    return 0;
}

