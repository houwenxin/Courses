#include "stdafx.h"
#include <iostream>
#include "CLibArray.h"
using namespace std;


void CArray::_check_capacity(int minimal)
{
	if (minimal <= Capacity)
    {
        return; 
    }

	int capacity = int(Capacity * 2); 
	recap(capacity < minimal ? minimal : capacity); 
}
void CArray::recap(int capacity)
{
	if (capacity == Capacity)
    {
        return; 
    }

    int *buff       = new int[capacity]; 
	Capacity  = capacity; 
	Size      = capacity < Size ? capacity : Size; 
  
	memcpy(buff, Buff, Size * sizeof(int)); 
	delete []Buff; 
    
	Buff = buff; 
}
void CArray::append(int element)
{
	_check_capacity(Size+1);
	Buff[Size++] = element; 
}
int CArray::size()
{
	return Size;
}
int CArray::capacity()
{
	return Capacity;
}

int &CArray::at(int index)
{
	return Buff[index];
}
void CArray::copy(CArray &array)
{
	_check_capacity(array.Capacity); 
	memcpy(Buff, array.Buff, array.Size * sizeof(int)); 
	Size = array.Size;
}
bool CArray::compare(CArray &array)
{
	if (Size != array.Size)
    {
        return false; 
    }
    
	return memcmp(Buff, array.Buff, Size) == 0; 
}
void CArray::insert(int index, int element)
{
	_check_capacity(Size+1);
	for (int i = Size++; i > index; --i)
    {
		Buff[i] = Buff[i - 1];
    }

	Buff[index] = element;
}
