#pragma warning(disable:4996) 

#ifndef HUFFMANCODE_H_
#define HUFFMANCODE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0
typedef int Status;

typedef struct {
	unsigned long weight;
	unsigned int parent, lchild, rchild;
}HTNode, *HuffmanTree;
typedef char **HuffmanCode;

int FindMin(HuffmanTree &HT, int i) {
	int Min = 0;
	int MinWeight = 0;
	int j = 0;
	while (HT[j].parent != 0) j++;
	Min = j;
	MinWeight = HT[j].weight;
	for (j; j < i; j++ ) {
		if (HT[j].parent == 0 && HT[j].weight < MinWeight) {
			Min = j;
			MinWeight = HT[j].weight;
		}
	}
	HT[Min].parent = 1;    //Avoid being finded again.
	return Min;
}
Status Select(HuffmanTree &HT, int i, int &s1, int &s2) {
	s1 = FindMin(HT, i);
	s2 = FindMin(HT, i);
	return OK;
}
Status CreateHuffmanTree(HuffmanTree &HT, int *w, int n) { //w: weights
	int m = 2 * n - 1;
	HT = (HuffmanTree)malloc(sizeof(HTNode) * m);
	if (!HT) return ERROR;
	for (int i = 0; i < n; i++) {
		HT[i].parent = 0; //No parent
		HT[i].lchild = 0; //No children
		HT[i].rchild = 0;
		HT[i].weight = w[i];
	}
	for (int i = n; i < m; i++) {
		HT[i].parent = 0; //From n to m is Nothing
		HT[i].lchild = 0; 
		HT[i].rchild = 0;
		HT[i].weight = 0;
	}
	int s1, s2;
	for (int i = n; i < m; i++) {
		Select(HT, i, s1, s2);	//Find the 2 most small-weight numbers from 0 to i,(i starts from n)  
		HT[s1].parent = i;	//Create child tree.
		HT[s2].parent = i;
		HT[i].lchild = s1;
		HT[i].rchild = s2;
		HT[i].weight = HT[s1].weight + HT[s2].weight;
	}
	return OK;
}
Status HuffmanCoding(HuffmanTree &HT, HuffmanCode &HC, int n) {
	HC = (HuffmanCode)malloc(n * sizeof(char *));
	char *cd = (char *)malloc(n * sizeof(char));
	if (!cd) return ERROR;
	cd[n - 1] = '\0';
	
	int i = 0;
	for (i = 0; i < n; i++) {
		int start = n - 1;
		int parent = HT[i].parent;
		int current = i;
		while (parent != 0) {
			if (HT[parent].lchild == current) cd[--start] = '0'; //lchild = 0
			else cd[--start] = '1';
			current = parent;
			parent = HT[parent].parent;
		}
		HC[i] = (char *)malloc((n - start) * sizeof(char));
		strcpy(HC[i], &cd[start]);
	}
#if 0
	for (i = 0; i < n; i++) {
		printf("%d, Weight %d: %s\n", i, HT[i].weight, HC[i]);
	}
#endif
	free(cd);
	cd = NULL;
	return OK;
}

Status HuffmanDeCoding(const HuffmanTree &HT, char *CodedInput, int n, int *Output, int &OutputNum) {
	HuffmanTree p = HT + 2 * n - 1 - 1; // Point to the root of HT, equal to p = &HT[m-1]; (start from 0.)
	while (*CodedInput != '\0') {
		if (*CodedInput == '0' && p->lchild) p = HT + p->lchild; //HT[p->lchild];
		else if (*CodedInput == '1' && p->rchild) p = HT + p->rchild;
		if (!p->lchild && !p->rchild) { //Reach the top.
			if (*CodedInput == '0') *Output = (HT + p->parent)->lchild;
			else *Output = (HT + p->parent)->rchild;
			Output++;
			p = HT + 2 * n - 2;	// Return to root.	
			OutputNum++;
		}
		CodedInput++;
	}
	return OK;
}



#endif
