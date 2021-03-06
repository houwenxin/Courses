// HashTable.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "HashTable.h"

int main()
{
#if 0 // Basic
	HashTable H;
	int a[8] = { 1,3,9,11,13,15,17,21 };
	Create(H);
	for (int i = 0; i < 8; i++) {
		InsertHash(H, a[i]);
	}
	int p, c, SL;
	float ASL = 0;
	for (int i = 0; i < 8; i++) {
		SL = 0;
		c = 0;
		SearchHash(H, a[i], p, c, SL);
		ASL = ASL + SL;
		printf("position: %d, SL: %d\n", p, SL);
	}
	ASL = ASL / 8;
	printf("ASL: %f\n", ASL);
	Destroy(H);
    return 0;
#endif
#if 1 //Advanced
	HashTable H;
	int a[8] = { 1,3,9,11,13,15,17,21 };
	Create(H);
	for (int i = 0; i < 8; i++) {
		InsertHash(H, a[i]);
	}
	int p, SL;
	float ASL = 0;
	for (int i = 0; i < 8; i++) {
		SL = 0;
		SearchHash(H, a[i], p, SL);
		ASL = ASL + SL;
		printf("position: %d, SL: %d\n", p, SL);
	}
	ASL = ASL / 8;
	printf("ASL: %f\n", ASL);
	Destroy(H);
	return 0;
#endif
}

