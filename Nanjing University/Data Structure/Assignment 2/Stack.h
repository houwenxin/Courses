#ifndef STACK_H_
#define STACK_H_

typedef char ElemType;

//Problem one
typedef struct Node {
	ElemType Data;
	struct Node *Next;
}LinkStack;

LinkStack *CreateStack(void) {
	LinkStack *Top = (LinkStack *)malloc(sizeof(LinkStack));
	Top->Data = NULL;
	Top->Next = NULL;
	return Top;
}
void push(LinkStack *Top, ElemType Elem) {
	LinkStack *NewSpace = (LinkStack *)malloc(sizeof(LinkStack));
	NewSpace->Data = Elem;
	NewSpace->Next = Top->Next;
	Top->Next = NewSpace;
}
ElemType pop(LinkStack *Top) {
	if (Top->Next == NULL) {
		printf("Stack Empty !\n");
		return -1;
	}
	else {
		LinkStack *Temp = Top->Next;
		ElemType Elem = Temp->Data;
		Top->Next = Top->Next->Next;
		free(Temp);
		Temp = NULL;
		return Elem;
	}
}
void DeleteStack(LinkStack *Top) {
	ElemType Judge = 0;
	printf("Deleting Stack...\n");
	if (Top->Next == NULL) {
		free(Top);
		Top = NULL;
	}
	else {
		while (Judge != -1) {
			Judge = pop(Top);
		}
		free(Top);
		Top = NULL;
	}
}

//Problem 2
void SpecialPush(LinkStack *Top, ElemType Elem) {
	if (Top->Next == NULL) {
		push(Top, Elem);
	}
	else {
		if (Elem == Top->Next->Data) {
			pop(Top);
		}
		else push(Top, Elem);
	}	
}
void PrintStack(LinkStack *Top) {
	if (Top->Next == NULL) {
		printf("No Element !\n");
		return;
	}
	LinkStack *Scan = Top->Next;
	while (Scan->Next != NULL) {
		printf("%c\n", Scan->Data);
		Scan = Scan->Next;
	}
}

//Problem 3

struct ListNode
{
	ElemType data;
	ListNode *next;
};
struct List
{
	int cap;
	int size;
	ListNode *head;
	ListNode *tail;
};
List &CreateList(int length);
void append(List &L, ElemType elem);
void print(List &L);
void Delete_all(List &L);
List &CreateList(int length)
{
	List L;
	L.head = (ListNode *)malloc(sizeof(ListNode));
	ListNode *scan = L.head;
	L.tail = NULL;
	L.size = 0;
	L.cap = length;
	L.head->data = NULL;
	for (int i = 0; i<L.cap + 1; ++i)
	{
		scan->next = (ListNode *)malloc(sizeof(ListNode));
		scan = scan->next;
		scan->data = NULL;
	}
	scan->next = L.tail;
	return L;
}

void append(List &L, ElemType elem)
{
	ListNode *scan = L.head->next;
	while (scan->data != NULL && scan->next != NULL)
	{
		scan = scan->next;
	}
	if (scan->data == NULL)
	{
		scan->data = elem;
	}
	if (scan->data != NULL && scan->next == NULL)
	{
		scan->next = (ListNode *)malloc(sizeof(ListNode));
		scan = scan->next;
		scan->data = elem;
		scan->next = L.tail;
		L.cap++;
	}
	L.size++;
}
void print(List &L)
{
	int count = 0;
	ListNode *scan = L.head->next;
	while (count < L.size)
	{
		printf("%d\n", scan->data);
		scan = scan->next;
		count++;
	}
}
void Delete_all(List &L)
{
	ListNode *scan = L.head;
	ListNode *temp = scan;
	while (scan != NULL)
	{
		temp = scan->next;
		free(scan);
		scan = temp;
	}
}
#endif // !1
