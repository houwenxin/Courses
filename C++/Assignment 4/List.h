#ifndef LIST_H_
#define LIST_H_

struct ListNode
{
	int num;
	ListNode *next;
	ListNode *pre;
	int& data() const
	{
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
		len = 0;
		Head = new ListNode;
		End = new ListNode;
		Head ->next = End;
		End ->pre = Head;
		ListNode *current;
		for (len = 0, current = list.begin(); len < list.size(); current = list.next(current))
		{
			append(current->data());
		}
	}
	void append(int i);
	int size() const;
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

	class Iterator
	{
	public:
		ListNode *ptr;
		Iterator(ListNode *p = NULL)
		{
			ptr = p;
		}
		int &operator *()
		{
			return ptr -> num;
		}
		ListNode* operator -> ()
		{
			return ptr;
		}
		Iterator& operator ++()
		{
			ptr = ptr->next;
			return *this;
		}
		Iterator operator ++(int)
		{
			ListNode *tmp = ptr;
			ptr = ptr->next;
			return tmp; 
		}
		bool operator != (ListNode *input)
		{
			return this->ptr != input;
		}
	};
	class ConstIterator
	{
	public:
		ListNode *ptr;
		ConstIterator(ListNode *p = NULL)
		{
			ptr = p;
		}
		int &operator *() const
		{
			return ptr -> num;
		}
		ListNode* operator -> () const
		{
			return ptr;
		}
		ConstIterator& operator ++()
		{
			ptr = ptr->next;
			return *this;
		}
		ConstIterator operator ++(int)
		{
			ListNode *tmp = ptr;
			ptr = ptr->next;
			return tmp; 
		}
		bool operator != (ListNode *input)
		{
			return this->ptr != input;
		}
	};

	void remove(Iterator tmp);
	void insert(Iterator tmp, int num);
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
