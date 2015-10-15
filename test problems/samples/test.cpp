#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void cputest1(int N, double *p);
void gputest1(int N, double *p);

int main()
{
	int N;
	double cpu_t1, gpu_t1, *p, *q;
	clock_t t1, t2;
	
	p = (double*) malloc(1*sizeof(double));
	q = (double*) malloc(1*sizeof(double));
	
	printf("Input N = ");
	scanf("%d",&N);
	*p = 1.0;
	*q = 1.0;
	
	t1 = clock();
	cputest1(N, p);
	t2 = clock();
	cpu_t1 = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	t1 = clock();
	gputest1(N, q);
	t2 = clock();
	gpu_t1 = 1.0*(t2-t1)/CLOCKS_PER_SEC;

	printf("test1 cpu times = %f \n", cpu_t1);
	printf("test1 gpu times = %f \n", gpu_t1);
	printf(" p = %f \n q = %f \n", *p, *q);
	free(p);
	free(q);
	return 0;
}

void cputest1(int N, double *p)
{
	int i;
	for(i=0;i<N;i++)
	{
		*p = *p + 1;
	}
}

void gputest1(int N, double *q)
{
	int i;
	#pragma acc data copyout(q[0:1])
	#pragma acc kernels 
	for(i=0;i<N;i++)
	{
		*q = *q + 1;
	}
}
