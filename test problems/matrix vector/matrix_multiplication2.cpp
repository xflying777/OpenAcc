/* matrix-acc-func.c */
#include <stdio.h>
#include <stdlib.h>
#define SIZE 1000
 
int doTest(float *restrict a, float *restrict b, float *restrict c, int size)
{
	int i,j,k;
	
	#pragma acc kernels deviceptr(a, b) copyout(c[0:size*size-1]) 
	{
	// Initialize matrices.
	#pragma acc loop independent
	for (i = 0; i < size; ++i) 
	{
	#pragma acc loop independent
		for (j = 0; j < size; ++j) 
		{
			a[i*size+j] = (float)i + j;
			b[i*size+j] = (float)i - j;
			c[i*size+j] = 0.0f;
		}	
	}
	
	// Compute matrix multiplication.
	#pragma acc loop independent
	for (i = 0; i < size; ++i) 
	{
	#pragma acc loop independent
		for (j = 0; j < size; ++j) 
		{
	#pragma acc loop seq
			for (k = 0; k < size; ++k) 
			{
				c[i*size+j] += a[i*size+k] * b[k*size+j];
			}
		}
	}
}
}
	
int main()
{
	int i,j,k;
	int size=SIZE;
	float *a= (float*)malloc(sizeof(float)*size*size);
	float *b= (float*)malloc(sizeof(float)*size*size);
	float *c= (float*)malloc(sizeof(float)*size*size);
	
	
	doTest(a,b,c, size);
	
	free(a);
	free(b);
	
	// ****************
	// double-check the OpenACC result sequentially on the host
	// ****************
	float *seq= (float*)malloc(sizeof(float)*size*size);
	// Initialize the seq matrix
	for(i = 0; i < size; ++i) 
		for(j = 0; j < size; ++j) 
			seq[i*SIZE+j] = 0.f;
	
	// Perform the multiplication
	for (i = 0; i < size; ++i) 
		for (j = 0; j < size; ++j) 
			for (k = 0; k < size; ++k) 
				seq[i*size+j] += (i+k) * (k-j);
	
	// check all the OpenACC matrices
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			if(c[i*size+j] != seq[i*size+j]) 
			{
				printf("Error (%d %d) (%g, %g)\n", i,j, c[i*size+j], seq[i*size+j]);
				exit(1);
			}
	free(c);
	free(seq);
	
	printf("OpenACC matrix multiplication test was successful!\n");
	
	return 0;
}
