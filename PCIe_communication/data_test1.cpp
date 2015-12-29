#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/*
Test: Let c = (a+b) .* (a-b), and trying to do it by three step.
      step 1. x = a + b
      step 2. y = a - b
      step 3. c = x .* y
Goal: Trying to do it with three different steps on GPU.(Only once of data copy)
*/
void initial(float *data1, float *data2, int N);
float error(float *x, float *y, int N);
void operation_gpu(float *a, float *b, float *c, int N);
void operation_cpu(float *a, float *b, float *c, int N);

int main()
{
	int N;
	float *a, *b, *c_cpu, *c_gpu;
	clock_t t1, t2;
	
	printf(" Please input n = ");
	scanf("%d", &N);
	
	a = (float *)malloc(N*sizeof(float));
	b = (float *)malloc(N*sizeof(float));
	c_cpu = (float *)malloc(N*sizeof(float));
	c_gpu = (float *)malloc(N*sizeof(float));
	
	t1 = clock();
	operation_cpu(a, b, c_cpu, N);
	t2 = clock();
	printf(" cpu times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
	t1 = clock();
	operation_gpu(a, b, c_gpu, N);
	t2 = clock();
	printf(" gpu times = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
	printf(" error = %f \n", error(c_cpu, c_gpu, N));	
	
	return 0;
}

void initial(float *data1, float *data2, int N)
{
	int i;
	for (i=0; i<N; i++)
	{
		data1[i] = 1.0*i;
		data2[N-i] = 1.0*i;
	}
}

float error(float *x, float *y, int N)
{
	int i;
	float error, temp;
	error = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > error)	error = temp;
	}
	return error;
}

void operation_gpu(float *a, float *b, float *c, int N)
{
	int i;
	float *x, *y;
	x = (float *)malloc(N*sizeof(float));
	y = (float *)malloc(N*sizeof(float));
	#pragma acc data copyin(a[0:N], b[0:N]), copyout(c[0:N]), create(x[0:N], y[0:N])
	{
		#pragma acc parallel loop independent
		// x = a + b
		for (i=0; i<N; i++)
		{
			x[i] = a[i] + b[i];
		}
		
		#pragma acc parallel loop independent
		// y = a - b
		for (i=0; i<N; i++)
		{
			y[i] = a[i] - b[i];
		}
		
		#pragma acc parallel loop independent
		// c = x .* y
		for (i=0; i<N; i++)
		{
			c[i] = x[i] * y[i];
		}
	}
}

void operation_cpu(float *a, float *b, float *c, int N)
{
	int i;
	float *x, *y;
	x = (float *)malloc(N*sizeof(float));
	y = (float *)malloc(N*sizeof(float));
	
	// x = a + b
	// y = a - b
	for (i=0; i<N; i++)
	{
		x[i] = a[i] + b[i];
		y[i] = a[i] - b[i];
	}
	
	// c = x .* y
	for (i=0; i<N; i++)
	{
		c[i] = x[i] * y[i];
	}
}
