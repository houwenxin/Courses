#include "stdafx.h"
#include "List.h"

void CPPList::append(int i)
{
	ListNode *scan;
	ListNode *pnew;
	scan = Head;
	pnew = new ListNode;
	pnew->num = i;
	pnew ->next = End;
	End ->pre = pnew;
	if(len == 0)
	{
		//printf("len = 0 here\n");
		Head = pnew;
	}
	else
	{
		//printf("len = %d here\n",CPPList::len);
		while(scan->next != End)
		{
			scan = scan->next;
		}
		scan->next = pnew;
		pnew->pre = scan;
	}
	len++;
}
int CPPList::size()
{
	return len;
}
void CPPList::remove(ListNode *tmp)
{	
	ListNode *scan = Head;
	if(scan == tmp)
	{
		//tmp->next->pre = NULL;
		Head = tmp->next;
		delete []tmp;
		tmp = NULL;
		len--;
	}
	else
	{
		while(scan ->next != End)
		{
			scan = scan->next;
			if(scan == tmp)
			{
				scan->pre->next = scan->next;
				scan->next->pre = scan->pre;
				scan->num = 0;
				delete []tmp;
				tmp = NULL;
				len--;
				break;
				printf("These words shouldn't appear in the program !");
			}
		}
	}
}

void CPPList::insert(ListNode *tmp, int num)
{
	ListNode *scan = Head;
	ListNode *pnew = new ListNode;
	pnew->num = num;
	if(scan == tmp || len == 0)
	{	
		pnew->pre = NULL;
		Head = pnew;
		if(len == 0){
			pnew->next = End;
			End->pre = pnew;
		}
		else{
			pnew->next = tmp;
			tmp->pre = pnew;
		}
		len++;
	}
	else
	{
		while(scan != End)
		{
			scan = scan->next;
			if(scan == tmp)
			{
#if 0
				if(len == 0)
				{
					pnew->pre = NULL;
					Head = pnew;
					pnew->next = End;
					End->pre = pnew;
				}
				else
				{
#endif
					pnew->next = scan;
					scan->pre->next = pnew;
					pnew->pre = scan->pre;
					scan->pre = pnew;
				
				len++;
				break;
			}
		}
	}
}

void CPPList::clear()
{
	ListNode *scan = Head;
	while(scan != End)
	{
		scan->num = 0;
		scan = scan->next;
		delete []scan->pre;
		scan->pre = NULL;
		len--;
	}
}