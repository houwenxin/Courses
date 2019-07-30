#ifndef CIRCULARQUEUE_H_
#define CIRCULARQUEUE_H_

#define MAXSIZE 101
#define ERROR -1
#define OVERFLOW -2
#define OK 0
typedef int ElemType;


typedef struct {
	ElemType *Data;
	int front;
	int rear;
}CirQueue;

int InitQueue(CirQueue &Q) {
	Q.Data = (ElemType *)malloc(MAXSIZE * sizeof(ElemType));
	if (Q.Data == NULL) {
		printf("Overflow !\n");
		exit(OVERFLOW);
	}
	Q.front = Q.rear = 0;
	return OK;
}
int QueueLength(CirQueue Q) {
	return (Q.rear - Q.front + MAXSIZE) % MAXSIZE;
}
void AddElem(CirQueue &Q, ElemType elem) {
	if ((Q.rear + 1) % MAXSIZE == Q.front) {
		printf("Queue is full !\n");
		return;
	}
	Q.Data[Q.rear] = elem;
	Q.rear = (Q.rear + 1) % MAXSIZE;
}
ElemType DeleteElem(CirQueue &Q) {
	if (Q.front == Q.rear) {
		printf("Queue Empty !\n");
		return ERROR;
	}
	else {
		ElemType Temp = Q.Data[Q.front];
		Q.front = (Q.front + 1) % MAXSIZE;
		return Temp;
	}
}
ElemType DeleteRearElem(CirQueue &Q) {
	if (Q.front == Q.rear) {
		printf("Queue Empty !\n");
		return ERROR;
	}
	else {
		ElemType Temp = (Q.rear - 1 + MAXSIZE) % MAXSIZE;
		Q.rear = (Q.rear - 1 + MAXSIZE) % MAXSIZE;
		return Temp;
	}
}
void DeleteQueue(CirQueue &Q) {
	Q.front = 0;
	Q.rear = 0;
	free(Q.Data);
	Q.Data = NULL;
}
void PrintQueue(CirQueue Q) {
	for (int i = Q.front; i < (Q.rear +MAXSIZE) % MAXSIZE; i++) {
		printf("%d\n", Q.Data[i]);
	}
}
#endif