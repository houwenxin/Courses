clc; clear all;

A = [1/2; 1/2];
B = [1/4; 3/4];
C = [1/8; 7/8];
KLab = KL(A,B)
KLba = KL(B,A)
KLac = KL(A,C)
KLca = KL(C,A)
KLbc = KL(B,C)
KLcb = KL(C,B)
KLaa = KL(A,A)
KLbb = KL(B,B)
KLcc = KL(C,C)