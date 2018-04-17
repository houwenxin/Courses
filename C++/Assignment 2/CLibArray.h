#ifndef __CLIBARRAY__
#define __CLIBARRAY__

class CArray
{
public:
	CArray()
	{
		Size = 0;
		Buff = new int[10];
		Capacity = 10;
	}
	void append(int element);
	void recap(int capacity);
	int size();
	int capacity();
	int &at(int index);
	void copy(CArray &array);
	bool compare(CArray &array);
	void insert(int index, int element);
	~CArray()
	{
		Size = 0;
		delete []Buff;
		Buff = NULL;
		Capacity = 0;
	}
private:
	int Size;
	int *Buff;
	int Capacity;
	void _check_capacity(int minimal);
};

#endif