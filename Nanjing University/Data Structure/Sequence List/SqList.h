#ifndef SQLIST_H_
#define SQLIST_H_

#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INIT_SIZE 10
#define INCREMENT_SIZE 5

typedef int Status;
typedef int ElemType;

typedef struct {
	ElemType *elem;
	int length;
	int size;
}SqList;

// Initialize
Status InitList(SqList *L){
	L->elem = (ElemType *)malloc(INIT_SIZE * sizeof(ElemType));
	if(!L->elem){	//L->elem == NULL
		return ERROR;
	}//Check
	L->length = 0;
	L->size = INIT_SIZE;
	return OK;
}

//Destroy
Status DestroyList(SqList *L){
	free(L->elem);
	L->length = 0;
	L->size = 0;
	return OK;
}

//Clear (Note: Not Destroy)
Status ClearList(SqList *L){
	L->length = 0;
	return OK;
}

//Judge if the list is empty
Status isEmpty(const SqList L){
	if(L.length == 0){
		return TRUE;
	}
	else{
		return FALSE;
	}
}

//Get List's length
Status getLength(const SqList L){
	return L.length;
}

//Get element from its location (start from 1)
Status GetElem(const SqList L, int i, ElemType *e){
	if(i < 1 || i > L.length){
		return ERROR;
	}
	*e = L.elem[i-1];
	return OK;
}

//Additional: compare if two elements are equal
Status compare(ElemType e1, ElemType e2){
	if(e1 == e2){
		return 0;
	}
	else if(e1 > e2){
		return 1;
	}
	else {
	return -1;
	}
}

//Find a number in the list
Status FindElem(const SqList L, ElemType e, Status (*compare)(ElemType, ElemType)){
	int i;
	for(i = 0; i < L.length; i++){
		if(!(*compare)(L.elem[i], e)){	//if L.elem[i] == e, return its location (+1)
			return i+1;
		}
	}
	if(i >= L.length){
		return ERROR;
	}
}

//Given a number, find the previous one
Status PreElem(const SqList L, ElemType e, ElemType *pre_e){
	int i;
	for(i = 0; i < L.length; i++){
		if(e == L.elem[i]){	//find the given number in the list
			if(i != 0){
				*pre_e = L.elem[i-1];
			}
			else {
				return ERROR;
			}
		}
	}
	if(i >= L.length){
		return ERROR;
	}
}

//Given a number, find the next one
Status NextElem(const SqList L, ElemType e, ElemType *next_e){
	int i;
	for(i = 0; i < L.length; i++){
		if(e == L.elem[i]){
			if(i < L.length - 1){
				*next_e = L.elem[i+1];
				return OK;
			}
			else {
				return ERROR;
			}
		}
	}
	if(i >= L.length){
		return ERROR;
	}
}
		
//Insert Element
Status InsertElem(SqList *L, int i, ElemType e){
	ElemType *new_elem;
	if(i < 1 || i > L->length + 1){
		return ERROR;
	}
	if(L->length >= L->size){ //Make sure there is space to add new element
		new_elem = (ElemType *)realloc(L->elem, (L->size + INCREMENT_SIZE) * sizeof(ElemType));
		if(!new_elem){
			return ERROR;
		}
		L->elem = new_elem;
		L->size += INCREMENT_SIZE;
	}
	ElemType *p = &L->elem[i-1];
	ElemType *q = &L->elem[L->length-1]; //length < size
	for(; q >= p; q--){
		*(q+1) = *q;
	}
	*p = e;
	L->length++;
	return OK;
}

//Delete an element
Status DeleteElem(SqList *L, int i, ElemType *e){
	if(i < 1 || i > L->length){
		return ERROR;
	}
	ElemType *p = &L->elem[i-1];
	*e = *p;
	for(; p < &L->elem[L->length]; p++){
		*(p) = *(p+1);
	}
	L->length--;
	return OK;
}
//Visit Element and operate, print here
void visit(ElemType e){
	printf("%d ", e);
}
//Traverse the list
Status TraverseList(const SqList L, void (*visit)(ElemType)){
	int i;
	for(i = 0; i < L.length; i++){
		visit(L.elem[i]);
	}
	return OK;
}

#endif
