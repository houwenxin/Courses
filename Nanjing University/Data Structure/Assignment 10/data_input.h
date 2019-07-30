/*
* MIT License
*
* Copyright (c) 2017 wen.gu <454727014@qq.com>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef __DATA_INPUT_H__
#define __DATA_INPUT_H__

#include <fstream>

using namespace std;

#define GRB4(a)  ((unsigned int)((((unsigned char*)(a))[0] << 24) | (((unsigned char*)(a))[1] << 16) |  \
		        (((unsigned char*)(a))[2] <<  8) | ((unsigned char*)(a))[3]))

#define DATA_INPUT_IMAGE_FLAG 0x00000803

#define DATA_INPUT_LABEL_FLAG 0x00000801

class dataInput
{
public:
	dataInput();
	~dataInput();
public:
	void reset();
	bool openLabelFile(const char* url);
	bool openImageFile(const char* url);

	bool readIndex(int* label);
	bool readImage(char imageBuf[]);

	bool read(int* label, char imageBuf[]);

	inline int numLable() { return mNumLabel; }
	inline int numImage() { return mNumImage; }

	inline int labelLength() { return mLabelLen; }
	inline int imageLength() { return mImageLen; }

	inline int imageWidth() { return mImageWidth; }
	inline int imageHeight() { return mImageHeight; }
private:
	int mNumLabel;
	int mNumImage;

	int mLabelLen;
	int mImageLen;
	int mImageWidth;
	int mImageHeight;
	int mImageStartPos;
	int mLableStartPos;
	fstream mLabelFile;
	fstream mImageFile;
};

#endif // !__DATA_INPUT_H__

