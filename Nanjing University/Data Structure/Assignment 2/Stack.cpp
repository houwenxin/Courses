// Stack.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <malloc.h>
#include "Stack.h"


int main()
{
#if 0
	LinkStack *NewStack = CreateStack();
	ElemType Elem = NULL;
	ElemType PopElem = NULL;
	printf("Enter your stack elements， enter $ as the end: \n");
	Elem = getchar();
	while (Elem != '$') {
		SpecialPush(NewStack, Elem);
		Elem = getchar();
	}
	printf("Popping:\n");
	while (PopElem != -1) {
		PopElem = pop(NewStack);
		if(PopElem != -1) printf("%c\n", PopElem);
	}
	DeleteStack(NewStack);
#endif

#if 1
	List L = CreateList(200);
	LinkStack *NewStack = CreateStack();
	for (int i = 3; i <= 100; i++)
	{
		append(L, i);
	}
	//print(L);
	ListNode *scan = L.head->next;
	int count = 0;
	while (count < L.size)
	{
		push(NewStack, scan->data);
		//printf("%d\n", scan->data);
		scan = scan->next;
		count++;
	}
	List LReverse = CreateList(L.size);
	ElemType PopElem = NULL;
	while (PopElem != -1) {
		PopElem = pop(NewStack);
		if(PopElem != -1) append(LReverse, PopElem);
	}
	print(LReverse);
	DeleteStack(NewStack);
	Delete_all(L);
	Delete_all(LReverse);
#endif
    return 0;
}

