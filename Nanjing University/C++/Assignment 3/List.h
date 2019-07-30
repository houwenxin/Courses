#ifndef LIST_H_
#define LIST_H_

struct ListNode
{
	int num;
	ListNode *next;
	ListNode *pre;
	int& data() const
	{
		//return num;
		return const_cast<int &>(num);
	}
};
class CPPList
{
private:
	int len;
	ListNode *Head;
	ListNode *End;
	
public:
	CPPList()
	{
		len = 0;
		Head = new ListNode;
		End = new ListNode;
		Head ->next = End;
		End ->pre = Head;
	}
	CPPList(const CPPList &list)
	{
		//*this = list;
		len = 0;
		Head = new ListNode;
		End = new ListNode;
		Head ->next = End;
		End ->pre = Head;
		ListNode *current;
		for (len = 0, current = list.begin(); len < list.size(); current = list.next(current))
		{
			//printf("len:%d\n",len);
			//printf("current data:%d\n",current->data());
			append(current->data());
		}
		//printf("len:%d\n",len);
	}
	void append(int i);
	int size();
	inline ListNode *begin() const
	{
		return Head;
	}
	inline ListNode *end() const
	{
		return End;
	}
	inline ListNode *next(const ListNode *p) const
	{
		return p->next;
	}
	void remove(ListNode *tmp);
	void insert(ListNode *tmp, int num);
	void clear();
	~CPPList()
	{
		len = 0;
		delete Head;
		Head = NULL;
		delete End;
		End = NULL;
	}
};
























#endif