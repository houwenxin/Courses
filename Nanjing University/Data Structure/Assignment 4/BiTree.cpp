#include "stdafx.h"
#include "BiTree.h"

Status CreateBiTree(BiTree *T)
{
	ElemType e;
	scanf_s("%c", &e, 1);
	getchar();
	if (e == '*')
	{
		*T = NULL;
	}
	else
	{
		*T = (BiTree)malloc(sizeof(BiTNode));
		if (!T)
		{
			exit(OVERFLOW);
		}
		(*T)->data = e;
		CreateBiTree(&(*T)->lchild);    //创建左子树
		CreateBiTree(&(*T)->rchild);    //创建右子树
	}
	return OK;
}
Status Visit(ElemType e) {
	printf("%c ", e);
	return OK;
}
Status PreOrderTraverse(BiTree T, Status(*Visit)(ElemType e)) {		//DLR
	if (T) {
		Visit(T->data);
		PreOrderTraverse(T->lchild, Visit);
		PreOrderTraverse(T->rchild, Visit);
	}
	else printf("* ");
	return OK;
}
Status InOrderTraverse(BiTree T, Status(*Visit)(ElemType e)) {		//LDR
	if (T) {
		InOrderTraverse(T->lchild, Visit);
		Visit(T->data);
		InOrderTraverse(T->rchild, Visit);
	}
	else printf("* ");
	return OK;
}
Status PostOrderTraverse(BiTree T, Status(*Visit)(ElemType e)) {	//LRD
	if (T) {
		PostOrderTraverse(T->lchild, Visit);
		PostOrderTraverse(T->rchild, Visit);
		Visit(T->data);
	}
	else printf("* ");
	return OK;
}
Status ExchangeBiTnode(BiTree *T) {
	if (!(*T) || !(*T)->lchild && !(*T)->rchild) {
		return OK;
	}
	BiTNode *TempNode = (*T)->lchild;
	(*T)->lchild = (*T)->rchild;
	(*T)->rchild = TempNode;
	if ((*T)->lchild) {
		ExchangeBiTnode(&(*T)->lchild);
	}
	if ((*T)->rchild) {
		ExchangeBiTnode(&(*T)->rchild);
	}
}