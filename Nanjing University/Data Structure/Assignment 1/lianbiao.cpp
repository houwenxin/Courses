# include <stdio.h>
# include <malloc.h>
# include "lianbiao.h"

List &CreateList(int length)
{
	List L;
	L.head = (ListNode *)malloc(sizeof(ListNode));
	ListNode *scan = L.head;
	L.tail = NULL;
	L.size = 0;
	L.cap = length;
	L.head->data = NULL;
	for(int i = 0; i<L.cap + 1;++i)
	{
		scan->next = (ListNode *)malloc(sizeof(ListNode));
		scan = scan->next;
		scan ->data = NULL;
	}
	scan->next = L.tail;
	return L;
}

void append(List &L, ElemType elem)
{
	ListNode *scan = L.head->next;
	while(scan ->data != NULL && scan -> next != NULL)
	{
		scan = scan ->next;
	}
	if(scan ->data == NULL)
	{
		scan->data = elem;
	}
	if(scan ->data != NULL && scan -> next == NULL)
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
ListNode *find_pre_location(List &L, int i)
{
	if (i > 1)
	{
		int count = 1;
		ListNode *scan = L.head->next;
		for (count = 1; count < i - 1; count++)
		{
			scan = scan->next;
		}
		return scan;
	}
	else if (i == 1)
	{
		return L.head;
	}
}
void Insert(List &L, int i, ElemType e)
{
	ListNode *scan = L.head->next;
	ListNode *loca;
	ListNode *goal_loca = find_pre_location(L, i);
	if (L.cap > L.size)
	{
		while (scan->next->data != NULL)
		{
			scan = scan->next;
		}
		scan->next->data = e;
		loca = scan->next;
		scan->next = scan->next->next;
		loca->next = goal_loca->next;
		goal_loca->next = loca;
		L.size++;
	}
	else if (L.cap == L.size)
	{
		ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
		new_node->data = e;
		new_node->next = goal_loca->next;
		goal_loca->next = new_node;
		L.size++;
		L.cap++;
	}
}
void Delete(List &L, int i, ElemType e)
{
	int count = 1;
	ListNode *scan = find_pre_location(L, i);
	scan->next->data = NULL;
	scan->next = scan->next->next;
	ListNode *supple = (ListNode *)malloc(sizeof(ListNode));
	supple->data = NULL;
	ListNode *find_tail = L.head->next;
	while (find_tail->next != L.tail)
	{
		find_tail = find_tail->next;
	}
	find_tail->next = supple;
	supple->next = L.tail;
	//printf("Deleted number: %d\n", scan->next->data);
	L.size--;
	//L.cap--;
}

void Reverse(List &L)
{
	ListNode *q;
	ListNode *p = L.head->next;
	L.head->next = NULL;
	//while (count < L.size)
	while (p != NULL && p->next != NULL && p->data != NULL) 
	{
		//printf("%d\n", p->data);
		q = p;
		p = p->next;
		q->next = L.head->next;
		L.head->next = q;
	}
}
ListNode* FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2)
{
	ListNode *CommonNode;
	ListNode *scan1 = pHead1;
	ListNode *scan2 = pHead2;
	int distance;
	int i = 0;
	int size1 = 0;
	int size2 = 0;
	while (scan1->next != NULL && scan1->next->data != NULL)
	{
		scan1 = scan1->next;
		size1++;
		//printf("size1 %d\n", size1);
	}
	while (scan2->next != NULL && scan2->next->data != NULL)
	{
		scan2 = scan2->next;
		size2++;
	}
	//printf("size1 %d\n", size1);
	//printf("size2 %d\n", size2);

	scan1 = pHead1->next;
	scan2 = pHead2->next;
	if (size1 >= size2)
	{
		distance = size1 - size2;
		while (i < distance)
		{
			scan1 = scan1->next;
			i++;
		}
		while (scan1->data != scan2->data)
		{
			if (scan1->next == NULL || scan2->next == NULL)
			{
				break;
			}
			scan1 = scan1->next;
			scan2 = scan2->next;
		}
		CommonNode = scan1;
	}
	else if (size2 > size1)
	{
		distance = size2 - size1;
		while (i < distance)
		{
			scan1 = scan2->next;
			i++;
		}
		while (scan2->data != scan1->data)
		{
			if (scan2->next == NULL || scan1->next == NULL)
			{
				break;
			}
			scan1 = scan1->next;
			scan2 = scan2->next;
		}
		CommonNode = scan1;
	}
	return CommonNode;
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
void concate(List &L1, List &L2)
{
	ListNode *p = L2.head->next;
	while (p != NULL && p->next != NULL &&p->data!=NULL)
	{
		//printf("%d\n", p->data);
		p = p->next;
	}
	//printf("end\n%d\n",L1.head->next->data);
	p->next = L1.head->next;
	L1.head->next = L2.head->next;
	//return 
}