#ifndef LIANBIAO_H_
#define LIANBIAO_H_

typedef int ElemType;

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
void Insert(List &L, int i, ElemType e);
List &CreateList(int length);
void append(List &L, ElemType elem);
void print(List &L);
void Delete(List &L, int i, ElemType e);
ListNode *find_pre_location(List &L, int i);
void Reverse(List &L);
ListNode* FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2);
void Delete_all(List &L);
void concate(List &L1, List &L2);
#endif