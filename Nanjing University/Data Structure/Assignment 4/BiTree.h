#ifndef BITREE_H_
#define BITREE_H_

#include <stdio.h>
#include <stdlib.h>

#define OK 1
#define OVERFLOW -2

typedef char ElemType;
typedef int Status;

typedef struct BiTNode {
	ElemType data;
	struct BiTNode *lchild, *rchild;
}BiTNode, *BiTree;

Status CreateBiTree(BiTree *T);
Status Visit(ElemType e);
Status PreOrderTraverse(BiTree T, Status (*Visit)(ElemType e));
Status InOrderTraverse(BiTree T, Status (*Visit)(ElemType e));
Status PostOrderTraverse(BiTree T, Status (*Visit)(ElemType e));
Status ExchangeBiTnode(BiTree *T);
#endif