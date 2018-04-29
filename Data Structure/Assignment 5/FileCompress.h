#ifndef FILECOMPRESS_H_
#define FILECOMPRESS_H_

#include "HuffmanCode.h"

Status Compress(const char *FileName) {
	printf("Compressing File: %s...\n", FileName);
	int n = 256;  //Use ASCII here = 256
	int m = 2 * n - 1;
	int w[256] = { 0 };
	int BitsCount = 0;
	//Read all the characters and calculate the weight.
	int a;
	FILE *pRead = fopen(FileName, "rb"); //Binary Read
	if (!pRead) {
		printf("Error: file does not exist.\n");
		return ERROR;
	}
	a = fgetc(pRead);
	while (!feof(pRead)) {
		w[a]++;
		BitsCount++;
		a = fgetc(pRead);  //Should put this after w[a]++ to avoid EOF
	}
	fclose(pRead);

	//Build Huffman Tree and figure out Huffman code.
	HuffmanTree HT;
	HuffmanCode HC;
	CreateHuffmanTree(HT, w, n);
	HuffmanCoding(HT, HC, n);

	//Write the compressed file  test.txt
	char OutPutFileName[100] = "Zip";
	int NameCount = 3;
	for (int i = 0; FileName[i] != '.'; i++) {
		OutPutFileName[i+3] = FileName[i];
		NameCount++;
	}
	OutPutFileName[NameCount] = '.';
	OutPutFileName[NameCount + 1] = 't';
	OutPutFileName[NameCount + 2] = 'x';
	OutPutFileName[NameCount + 3] = 't';
	OutPutFileName[NameCount + 4] = '\0';
	
	FILE *pWrite = fopen(OutPutFileName, "wb");
	//Write the Number of bits.
	fwrite(&BitsCount, 2, 1, pWrite);
	//Record Source file's format
	for (int i = NameCount-3; i <= NameCount; i++) {
		fwrite(&FileName[i], 1, 1, pWrite);
		//printf("%c", FileName[i]);
	}
	//Record the Huffman Tree for uncompress.
	for (int i = n; i < m; i++) {
		fwrite(&HT[i].lchild, 2, 1, pWrite);
		fwrite(&HT[i].rchild, 2, 1, pWrite);
	}
	//Begin to Compress.
	pRead = fopen(FileName, "rb");
	int b_int = 0;
	HuffmanCode hc = HC;
	a = fgetc(pRead);
	char *p = hc[a]; // Equal to char *p = *(hc+a);
	while (!feof(pRead) || *p != '\0') {
		for (int i = 0; i < 8; i++) {   //256 = 2^8
			if (*p == '\0') {
				a = fgetc(pRead);
				if (a >= 0) p = hc[a];
				else {  //The a is the last one, add signs for decoding.
					fputc(253, pWrite);
					fputc(254, pWrite);
					fwrite(&i, 1, 1, pWrite);
					fputc('#', pWrite);
					break;	//Equals add 0 in the end
				}
			}
			int BinToDec = 1;
			for (int j = 0; j < 7 - i; j++) BinToDec = BinToDec * 2;
			b_int = b_int + (*p - '0') * BinToDec;
			p++;
		}
		fwrite(&b_int, 1, 1, pWrite);
		b_int = 0; //Another read.
	}
	fclose(pRead);
	fclose(pWrite);
	printf("Successfully Compressed !\n");
	printf("Compressed File Name: %s\n", OutPutFileName);
	return OK;
}

Status UnCompress(const char *FileName) {
	printf("Uncompressing File: %s...\n", FileName);
	int a;
	int n = 256;
	int m = 2 * n - 1;
	int BitsCount = 0;
	//Firt get the uncompressed file name.
	FILE *pRead = fopen(FileName, "rb");
	if (!pRead) {
		printf("Error: file does not exist.\n");
		return ERROR;
	}
	//Get the number of output needed.
	fread(&BitsCount, 2, 1, pRead);

	char OutPutFileName[100] = "Un";
	int NameCount = 2;
	for (int i = 0; FileName[i] != '.'; i++) {
		OutPutFileName[NameCount++] = FileName[i];
	}
	for (int j = 0; j < 4; j++) //.txt
	{
		a = fgetc(pRead);
		OutPutFileName[NameCount++] = a;
	}

	//Recover the Huffman Tree from the compressed file.
	HuffmanTree HT = (HuffmanTree) malloc(sizeof(HTNode) * m);
	for (int i = 0; i < n; i++) {
		HT[i].lchild = 0;
		HT[i].rchild = 0;
		HT[i].weight = 0;
	}
	for (int i = n; i < m; i++) {
		HT[i].weight = 0;
		fread(&a, 2, 1, pRead); HT[i].lchild = a; HT[a].parent = i; //Reconstructing
		fread(&a, 2, 1, pRead);	HT[i].rchild = a; HT[a].parent = i;
	}
	HT[m - 1].parent = 0; //Root Node.
	
	//Get the Coded Input for UnCompress().
	char Binary[8];
	char *CodedInput = (char *)malloc(sizeof(int) * BitsCount * 30); //I actually don't know why 30. But this works, at least for my test example.
	int j = 0;
	int a_check1 = 0;
	int a_check2 = 0;
	a = fgetc(pRead);
	while (!feof(pRead) && (a != 254 || a_check1 != 253)) {
		a_check1 = a;
		a_check2 = a_check1;
		for (int i = 0; i < 8; i++) {
			if (a) {
				Binary[7 - i] = '0' + a % 2;
				a = a / 2;
			}
			else Binary[7 - i] = '0';
		}
		for (int i = 0; i < 8; i++) {
			CodedInput[8 * j + i] = Binary[i]; //Unknown memory problem. Stack around variable Binary is corrupted ?
		}
		j++;
		a = fgetc(pRead);
	}
	
	int temp = 0;
	int LastNumBits = 0;
	if (a == 254) { //sign = 254.
		a = fgetc(pRead);
		LastNumBits = a;
		a = fgetc(pRead);
	}

	a = fgetc(pRead);
	for (temp = 0; temp < 8 - LastNumBits; temp++) a = a / 2;
	for (temp = 1; temp < LastNumBits+1; temp++) {				//Maybe I can consider a better way here.
		CodedInput[8 * j - 8 * 1 - temp + LastNumBits] = '0' + a % 2;
		a = a / 2;
	}
	CodedInput[8 * j - 8 * 1 + LastNumBits] = '\0';   //Minus 8 because we need to delete wrong decoding from the first sign.

	//Huffman Decoding.
	int *Output = (int *) malloc(sizeof(int) * BitsCount * 30);
	int OutputNum = 0;
	HuffmanDeCoding(HT, CodedInput, n, Output, OutputNum);

	FILE *pWrite = fopen(OutPutFileName, "wb");
	for (int i = 0; i < OutputNum; i++) {
		//printf("%c",Output[i]);
		fwrite(&Output[i], 1, 1, pWrite);
	}
	free(Output); Output = NULL;
	free(CodedInput); CodedInput = NULL;
	fclose(pRead);
	fclose(pWrite);
	printf("Successfully Uncompressed !\n");
	printf("Uncompressed File Name: %s\n", OutPutFileName);
	return OK;
}

Status Menu() {
	char key = 0;
	char FileName[100];
	printf("File Compression with Huffman Coding\n");
	printf("Do you want to ? 1. Compressing.\n");
	printf("                 2. Uncompressing.\n");
	printf("                 Q. Exit.\n");
	while (key != 'Q')
	{
		scanf("%c", &key);
		getchar();
		if (key == '1') {
			printf("Please enter your file name with format: ");
			scanf("%[^\n]", FileName); //scanf that can get string with space ' '.
			Compress(FileName);
			break;
		}
		else if (key == '2') {
			printf("Please enter your file name with format: ");
			scanf("%[^\n]", FileName);
			UnCompress(FileName);
			break;
		}
		else if (key == 'Q') {
			break;
		}
		else {
			printf("Only 1, 2, Q are allowed to enter.\n");
			continue;
		}
	}
	return OK;
}


#endif