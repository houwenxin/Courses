#ifndef FIND_H_
#define FIND_H_

using std::list;
using std::vector;

template <class T> 
typename list<T>::iterator myfind(typename list<T>::iterator iter1, typename list<T>::iterator iter2, T element)
{
	list<T>::iterator iter;
	for(iter = iter1; iter != iter2; ++iter)
	{
		if(*iter == element)
		{
			break;
		}
	}
	return iter;
}

template <class T>
typename vector<T>::iterator myfind(typename vector<T>::iterator iter1, typename vector<T>::iterator iter2, T element)
{
	vector<T>::iterator iter;
	for(iter = iter1; iter != iter2; ++iter)
	{
		if(*iter == element)
		{
			break;
		}
	}
	return iter;
}

#endif