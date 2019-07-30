#pragma once
#ifndef HASHTABLE_H_
#define HASHTABLE_H_

#include <stdio.h>
#include <stdlib.h>

#define SUCCESS 1
#define UNSUCCESS 0
#define DUPLICATE -1
#define MAXSIZE 100
#define OK 1

typedef int ElemType;
typedef int Status;

int Hash(int key) {
	int hash = key % 10;
	return hash;
}
#if 0
typedef struct {
	ElemType *elem;
	int count;
	int hashsize;
}HashTable;
Status Create(HashTable &H) {
	H.hashsize = MAXSIZE;
	H.count = 0;
	H.elem = (ElemType *)malloc(sizeof(ElemType)*H.hashsize);
	for (int i = 0; i < H.hashsize; i++) {
		H.elem[i] = -1;
	}
	return OK;
}
Status Destroy(HashTable &H) {
	H.hashsize = H.count = 0;
	free(H.elem);
	H.elem = NULL;
	return OK;
}

int collision(int &p, int &c) {
	p = (p + c) % MAXSIZE;
	return p;
}
Status SearchHash(HashTable H, ElemType key, int &p, int &c, int &SL) { // p: position, c: collision
	p = Hash(key);
	SL++;
	while (H.elem[p] != -1 && key != H.elem[p]) {
		collision(p, ++c);
		SL++;
	}
	if (key == H.elem[p])
		return SUCCESS;
	else return UNSUCCESS;
}
Status InsertHash(HashTable &H, ElemType e) {
	int c = 0;
	int p, SL;
	if (SearchHash(H, e, p, c, SL))
		return DUPLICATE;
	else if (c < H.hashsize - 1) {
		H.elem[p] = e;
		++H.count;
		return OK;
	}
}
#endif
#if 1
typedef struct node{
	ElemType elem;
	struct node *next;
} Node;
typedef struct HashNode{
	Node *data[MAXSIZE];
	int count;
	int hashsize;
}HashTable;

Status Create(HashTable &H) {
	H.hashsize = MAXSIZE;
	for (int i = 0; i < H.hashsize; i++) {
		H.data[i] = (Node *)malloc(sizeof(Node));
		H.data[i]->elem = -1;
		H.data[i]->next = NULL;
	}
	H.count = 0;
	return OK;
}
Status Destroy(HashTable &H) {
	for (int i = 0; i < H.hashsize; i++) {
		free(H.data[i]);
		H.data[i] = NULL;
	}
	H.count = H.hashsize = 0;
	return OK;
}
Status SearchHash(HashTable &H, ElemType key, int &p, int &SL) {
	p = Hash(key);
	Node *pnode = H.data[p];
	while (pnode != NULL){
		if (pnode->elem == key)
			return SUCCESS;
		pnode = pnode->next;
		SL++;
	}
	return UNSUCCESS;
}
Status InsertHash(HashTable &H, ElemType e) {
	int c = 0;
	int p, SL;
	if (SearchHash(H, e, p, SL))
		return DUPLICATE;
	Node *pnode = H.data[p];
	while (pnode->next != NULL) {
		pnode = pnode->next;
	}
	Node *newnode = (Node *)malloc(sizeof(Node));
	newnode->elem = e;
	newnode->next = NULL;
	pnode->next = newnode;
	return OK;
}
#endif
#endif