// BinarySearch.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>

typedef int ElemType;

typedef struct {
	ElemType *elem;
	int length;
} SSTable;	//Static Search Table
void Create(SSTable &ST, int n, ElemType *nums) {
	ST.length = n;
	ST.elem = (ElemType *)malloc(sizeof(ElemType) * ST.length);
	for (int i = 0; i < ST.length; i++) {
		ST.elem[i] = nums[i];
	}
}
void Destroy(SSTable &ST) {
	ST.length = 0;
	free(ST.elem);
	ST.elem = NULL;
}
int Binary_Search(SSTable ST, ElemType key, int &SL) {
	int low = 0;
	int high = ST.length - 1;
	int mid;
	while (low <= high) {
		mid = (low + high) / 2;
		SL++;
		if (key < ST.elem[mid]) {
			high = mid - 1;
		}
		else if (key > ST.elem[mid]) {
			low = mid + 1;
		}
		else {
			return mid + 1;
		}
	}
	return -1;
}
int main()
{
	SSTable ST;
	int a[8] = { 1,3,9,11,13,15,17,21 };
	float ASL = 0;
	Create(ST, 8, a);
	for (int i = 0; i < 8; i++) {
		int SL = 0;
		int location = Binary_Search(ST, a[i], SL);
		printf("Location: %d, SL:%d\n", location, SL);
		ASL = ASL + SL;
	}
	ASL = ASL / 8;
	printf("ASL: %f\n", ASL);
	Destroy(ST);
    return 0;
}

