
/// @brief [类的概要描述]
///
/// [描述你的类.]
///
/// @see 
///
template <class> class CPPListIterator;
template <class> class CPPListConstIterator;
template <class> class CPPList;
template <class T>
class ListNode
{
public:
	typedef T DataType; 

	ListNode() : _next(NULL), _prev(NULL) { }
	ListNode(DataType data) : _data(data), _next(NULL), _prev(NULL) { }

	DataType &data() { return _data; }

	const DataType data() const { return _data; }

protected:
	DataType _data; 
	ListNode *_next, *_prev; 

	friend class CPPList <T>; 
	friend class CPPListIterator<T>; 
	friend class CPPListConstIterator<T>; 

}; // class ListNode类定义结束.
template <class T>
class CPPListIterator
{
public:
	CPPListIterator() : _current(NULL) { }
	CPPListIterator(ListNode<T> *current) : _current(current) { }

	CPPListIterator &operator++()
	{
		_current = _current->_next; 
		return *this; 
	}

	const CPPListIterator operator++(int) 
	{
		CPPListIterator temp(*this); 
		_current = _current->_next; 
		return temp; 
	}

	bool operator==(const CPPListIterator &rhs) const
	{
		return rhs._current == this->_current; 
	}

	bool operator!=(const CPPListIterator &rhs) const
	{
		return !(*this == rhs); 
	}

	typename ListNode<T>::DataType &operator*() const
	{
		return _current->_data; 
	}

	typename ListNode<T>::DataType *operator->() const
	{
		return &(_current->_data); 
	}

private:
	ListNode<T> *_current; 
	friend class CPPList<T>; 
};

template <class T>
class CPPListConstIterator
{
public:
	CPPListConstIterator() : _current(NULL) { }
	CPPListConstIterator(ListNode<T> *current) : _current(current) { }

	CPPListConstIterator &operator++()
	{
		_current = _current->_next; 
		return *this; 
	}

	const CPPListConstIterator operator++(int) 
	{
		CPPListConstIterator temp(*this); 
		_current = _current->_next; 
		return temp; 
	}

	bool operator==(const CPPListConstIterator &rhs) const
	{
		return rhs._current == this->_current; 
	}

	bool operator!=(const CPPListConstIterator &rhs) const
	{
		return !(*this == rhs); 
	}

	typename const ListNode<T>::DataType &operator*() const
	{
		return _current->_data; 
	}

	typename const ListNode<T>::DataType *operator->() const
	{
		return &(_current->_data); 
	}

private:
	ListNode<T> *_current; 
};


/// @brief [类的概要描述]
///
/// [描述你的类.]
///
/// @see 
///
template <class T>
class CPPList
{
public:
	typename typedef ListNode<T>::DataType Element; 
	typename typedef CPPListIterator<T> Iterator; 
	typename typedef CPPListConstIterator<T> ConstIterator; 

	// CPPList类构造函数.
	CPPList() : _head(NULL), _tail(NULL), _size(0) { }
	CPPList(const CPPList &rhs); 
	CPPList &operator=(const CPPList &rhs); 


	// CPPList类析构函数
	~CPPList(); 

	// CPPList类接口.
public:
	// 列表是否为空
	bool is_empty() const { return _head == NULL; }

	// 列表元素个数
	int size() const { return _size; }

	// 返回list有效节点的起始位置begin及终止位置end
	// 这些节点应从begin开始，到end结束，但【不包括】end本身
	// 因此当list为空时，返回值应满足begin==end
	Iterator begin() { return Iterator(_head); }
	Iterator end()   { return Iterator(NULL); }  
	ConstIterator begin() const { return ConstIterator(_head); }
	ConstIterator end()   const { return ConstIterator(NULL); }  

	// 尾部追加数据
	void append(Element); 

	// 在current之前差远数据，应判断current有效性
	// 无效则无需动作
	void insert(CPPListIterator<T> current, Element); 

	// 删除current当前节点，应判断current有效性
	// 无效则无需动作
	void remove(CPPListIterator<T> current); 

	// 清空数据
	void clear(); 

	// CPPList类私有型成员变量.
private:
	///< 描述你的成员变量，如没有，请删除此行. 
	ListNode<T> *_head; 
	ListNode<T> *_tail; 
	int _size; 

}; // class CPPList类定义结束.




template <class T>
CPPList<T>::CPPList(const CPPList &rhs) : _head(NULL), _tail(NULL), _size(0)
{
	*this = rhs; 
}

template <class T>
CPPList<T> &CPPList<T>::operator=(const CPPList<T> &rhs)
{
	if (this == &rhs)
	{
		return *this; 
	}

	clear(); 

	ConstIterator current = rhs.begin(); 
	while (current != rhs.end())
	{
		append(*current); 
		++current; 
	}
	
	return *this; 
}

template <class T>
CPPList<T>::~CPPList() 
{
	clear(); 
}

template <class T>
void CPPList<T>::clear()
{
	Iterator current = begin();
	while (current != end())
	{
		remove(current++); 
	}
}

template <class T>
void CPPList<T>::append(Element data)
{
	ListNode<T> *node = new ListNode<T>(data); 

	if (is_empty())
	{
		_head = node; 
	}
	else {
		_tail->_next = node; 
		node->_prev = _tail; 
	}
	_tail = node; 

	++_size; 
}

template <class T>
void CPPList<T>::insert(Iterator iter, Element data)
{
	if (iter == end())
	{
		append(data); 
		return; 
	}

	ListNode<T> *current = iter._current; 
	ListNode<T> *node = new ListNode<T>(data); 

	if (_head == current)
	{
		_head = node; 
	}

	if (current->_prev != NULL)
	{
		current->_prev->_next = node; 
	}
	node->_prev = current->_prev;

	node->_next = current; 
	current->_prev = node; 

	++_size; 

}

template <class T>
void CPPList<T>::remove(Iterator iter)
{
	if (iter == end())
	{
		return; 
	}

	ListNode<T> *current = iter._current; 

	if (current->_prev != NULL)
	{
		current->_prev->_next = current->_next; 
	}
	if (current->_next != NULL)
	{
		current->_next->_prev = current->_prev; 
	} 

	if (_head == current)
	{
		_head = current->_next; 
	}

	if (_tail == current)
	{
		_tail = current->_prev; 
	}

	delete current; 
	--_size; 
}