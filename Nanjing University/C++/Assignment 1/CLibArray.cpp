#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "CLibArray.h"



void array_initial(CArray &array)
{
	CArray *ptr = &array;
	array.elem = NULL;//lazy evaluation
	//ptr->elem = (int *)malloc(sizeof(int));
	ptr->size = 0;
	ptr->cap = 0;
}
void array_recap(CArray &array, int cap)
{
	int *buff = new int[cap];
	CArray *ptr = &array;
	if(array.cap == cap)
	{
		return;
	}
	array.cap = cap;
	array.size = cap < array.size? cap:array.size;
	memcpy(buff,array.elem,array.size*sizeof(int));
	array.elem = buff;
	//ptr->cap = cap;
	//ptr->elem = (int *)realloc(ptr->elem,sizeof(int)*cap);
}
int array_capacity(CArray array)
{
	return array.cap;
}
void _check_capacity(CArray &array, int minimal)
{
	if(minimal <= array.cap)
	{
		return;
	}
	int capacity = int(array.cap * 2);
	array_recap(array,capacity<minimal?minimal:capacity);
}
void array_append(CArray &array, int i)
{
	CArray *ptr=&array;

#if 0
	if(ptr->size==ptr->cap)
	{
		ptr->cap++;
		ptr->elem = (int *)realloc(ptr->elem,sizeof(int)*ptr->cap);
	}
#endif

	_check_capacity(array, array.size+1);
	ptr->elem[ptr->size]=i;
	ptr->size++;
}
int array_size(CArray array)
{
	return array.size;
}

int &array_at(CArray &array, int i)
{
	CArray *ptr = &array;
	return ptr->elem[i];
}
void array_copy(CArray &array1, CArray &array2)
{
#if 0
	CArray *ptr1 = &array1;
	CArray *ptr2 = &array2;
	ptr2->cap = ptr1->cap;
	ptr2->size = ptr1->size;
	ptr2->elem =(int *)malloc(sizeof(int)*ptr2->cap);
	for (int i = 0; i < ptr2->size; ++i)
    {
		ptr2->elem[i]=ptr1->elem[i]; 
    }
#endif
#if 1
	_check_capacity(array2, array1.cap);
	memcpy(array2.elem,array1.elem,array1.size * sizeof(int));
	array2.size = array1.size;
#endif
}
bool array_compare(CArray &array1, CArray &array2)
{
	CArray *ptr1 = &array1;
	CArray *ptr2 = &array2;
	int count=0;
	if(ptr2->size == ptr1->size && ptr2->cap == ptr1->cap)
	{
		for (int i = 0; i < ptr2->size; ++i)
		{
			if(ptr2->elem[i] != ptr1->elem[i])
			{
				count++;
			}
		}
	}
	else
		count++;
	if(count == 0)
		return true;
	else
		return false;
}

void array_insert(CArray &array, int location, int number)
{
	CArray *ptr = &array;

#if 0
	if(ptr->size == ptr->cap)
	{
		ptr->cap++;
	}
	ptr->size++;
	ptr->elem = (int *)realloc(ptr->elem,sizeof(int)*ptr->cap);
	for (int i = ptr->size; i > location; --i)
    {
		ptr->elem[i]=ptr->elem[i-1]; 
    }
	ptr->elem[location] = number;
#endif

	_check_capacity(array, array.size+1);
	for (int i = ptr->size; i > location; --i)
    {
		ptr->elem[i]=ptr->elem[i-1]; 
    }
	ptr->elem[location] = number;

}
void array_destroy(CArray &array)
{
	CArray *ptr = &array;
	ptr->cap = 0;
	ptr->size = 0;
	free(ptr->elem);
	ptr->elem = NULL;
}