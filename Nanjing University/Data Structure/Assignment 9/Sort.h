#pragma once
#ifndef SORT_H_
#define SORT_H_

typedef int ElemType;
//Bubble Sort
void swap(ElemType &a, ElemType &b) {
	ElemType temp = b;
	b = a;
	a = temp;
}
void BubbleSort(ElemType *array, int length) {
	int i, j ,Sorted;
	for (i = 0; i < length; i++) {
		Sorted = 1;
		for (j = 0; j < length - i - 1; j++) {
			if (array[j] > array[j+1]) {
				swap(array[j], array[j+1]);
				Sorted = 0;
			}
		}
		if (Sorted == 1)	//To save unnecessary calculation.
			break;
	}
}

//Quick Sort
void QuickSort(ElemType *array, int low, int high){
	if (low >= high)	//To stop recursion.
		return;
	int first = low;
	int last = high;
	ElemType pivotkey = array[first];
	while (first < last) {
		while (first < last && array[last] >= pivotkey) {
			last--;
		}
		array[first] = array[last];
		while (first < last && array[first] <= pivotkey) {
			first++;
		}
		array[last] = array[first];
	}
	array[first] = pivotkey;
	QuickSort(array, low, first - 1);
	QuickSort(array, first + 1, high);
}

#endif