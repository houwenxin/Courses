#ifndef __CLIBARRAY__
#define __CLIBARRAY__


template <class T> 
class CArray
{
public:
	CArray(); 
	~CArray(); 

	int capacity() const
	{
		return _capacity; 
	}

	int size() const
	{
		return _size; 
	}

	void recap(int capacity); 

	T &at(int index)
	{
		return _buff[index]; 
	}

	T at(int index) const
	{
		return _buff[index]; 
	}

	void append(T element); 
	void insert(int index, T element); 

	void copy(const CArray &rhs); 
	bool compare(const CArray &rhs) const; 

private:
	void _check_capacity(int minimal); 

	T *_buff; 
	int _size; 
	int _capacity; 
};

template <class T>
CArray<T>::CArray() : _buff(NULL), _size(0), _capacity(0)
{
}

template <class T>
CArray<T>::~CArray()
{
    delete []_buff; 

    _buff      = NULL; 
    _capacity  = 0; 
    _size      = 0; 
}

template <class T>
void CArray<T>::recap(int cap)
{
    if (cap == _capacity)
    {
        return; 
    }
    _capacity  = cap; 
	
    T *buff = new T[_capacity]; 
    _size      = _capacity < _size ? _capacity : _size; 
    
    memcpy(buff, _buff, _size * sizeof(T)); 
    delete []_buff; 
    
    _buff = buff; 
}

template <class T>
void CArray<T>::_check_capacity(int minimal)
{
    if (minimal <= _capacity)
    {
        return; 
    }

	int capacity = int(_capacity * 2); 
    recap(capacity < minimal ? minimal : capacity); 
}

template <class T>
void CArray<T>::append(T element)
{
    _check_capacity(_size + 1); 

    _buff[_size++] = element ; 
}

template <class T>
void CArray<T>::insert(int index, T element)
{
    _check_capacity(_size + 1); 

    for (int i = _size; i > index; --i)
    {
        _buff[i] = _buff[i - 1];
    }

    _buff[index] = element; 
}

template <class T>
void CArray<T>::copy(const CArray &rhs) 
{
    _check_capacity(rhs._capacity); 

    memcpy(_buff, rhs._buff, rhs._size * sizeof(T)); 
    _size = rhs._size; 
}

template <class T>
bool CArray<T>::compare(const CArray &rhs) const
{
    if (rhs._size != _size)
    {
        return false; 
    }
    
    return memcmp(_buff, rhs._buff, _size * sizeof(T)) == 0; 
}

#endif