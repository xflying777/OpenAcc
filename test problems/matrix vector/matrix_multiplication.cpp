#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* matrix-acc-check.c */
#define SIZE 1500
float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];
float seq[SIZE][SIZE];
 
int main()
{
  int i,j,k;
  double t1, t2, gpu_time, cpu_time;
   
  // Initialize matrices.
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i][j] = (float)i + j;
      b[i][j] = (float)i - j;
      c[i][j] = 0.0f;
    }
  }
  
  t1 = clock(); 
  // Compute matrix multiplication.
#pragma acc kernels copyin(a,b) copy(c)
  for (i = 0; i < SIZE; ++i) 
    for (j = 0; j < SIZE; ++j) 
      for (k = 0; k < SIZE; ++k) 
    	c[i][j] = c[i][j] + a[i][k] * b[k][j];
 t2 = clock();
 gpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;
 
  // ****************
  // double-check the OpenACC result sequentially on the host
  // ****************
  // Initialize the seq matrix
  for(i = 0; i < SIZE; ++i) 
    for(j = 0; j < SIZE; ++j) 
      seq[i][j] = 0.0f;
   
   t1 = clock();
  // Perform the multiplication
  for (i = 0; i < SIZE; ++i) 
    for (j = 0; j < SIZE; ++j) 
      for (k = 0; k < SIZE; ++k) 
    	seq[i][j] = seq[i][j] + a[i][k] * b[k][j];
   t2 = clock();
   cpu_time = 1.0*(t2-t1)/CLOCKS_PER_SEC;
   
  // check all the OpenACC matrices
  for (i = 0; i < SIZE; ++i)
    for (j = 0; j < SIZE; ++j)
      if(c[i][j] != seq[i][j]) 
	  {
    	printf("Error %d %d\n", i,j);
      }
  printf("OpenACC matrix multiplication test was successful!\n");
  printf("gpu times = %f, cpu times = %f \n", gpu_time, cpu_time); 
  return 0;
}
