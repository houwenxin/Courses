// Find.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <assert.h>

#include <list>
#include <vector>
#include <iostream>
#include <algorithm>

#include "Find.h"

using std::cout;
using std::endl;
using std::list;
using std::vector; 

int _tmain(int argc, _TCHAR* argv[])
{	
	typedef list<int> IntList; 
	typedef vector<int> IntVector; 

	int array[] = {3, 4, 2, 1, 1, 2, 3, 5, 2};
	
	IntList ilist;
	IntVector ivector; 
	
	for (int i = 0; i < sizeof(array) / sizeof(int); ++i)
	{
		ilist.push_back(array[i]); 
		ivector.push_back(array[i]); 
	}
	ilist.sort(); 
	sort(ivector.begin(), ivector.end()); 

	for (int i = 0; i < sizeof(array) / sizeof(int); ++i)
	{
		IntList::iterator iter1 = myfind(ilist.begin(), ilist.end(), array[i]);
		assert(iter1 != ilist.end());
		assert(*iter1 == array[i]); 

		IntVector::iterator iter2 = myfind(ivector.begin(), ivector.end(), array[i]);
		assert(iter2 != ivector.end());
		assert(*iter2 == array[i]); 
	}

	IntList::iterator iter1 = myfind(ilist.begin(), ilist.end(), 22);
	assert(iter1 == ilist.end());

	IntVector::iterator iter2 = myfind(ivector.begin(), ivector.end(), 22);
	assert(iter2 == ivector.end());
	
	system("pause"); 
	return 0;
}

