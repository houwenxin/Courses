#ifndef CLIBARRAY_H_
#define CLIBARRAY_H_


struct CArray
{
	int cap;
	int size;
	int *elem;
};

void array_initial(CArray &array);
void array_recap(CArray &array, int cap);
int array_capacity(CArray array);
void array_append(CArray &array, int i);
int array_size(CArray array);
int &array_at(CArray &array, int i);
void array_copy(CArray &array1, CArray &array2);
bool array_compare(CArray &array1, CArray &array2);
void array_insert(CArray &array, int location, int number);
void array_destroy(CArray &array);



#endif