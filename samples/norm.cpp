#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double norm_cpu(double *x)
{
	int n, i;
	double norm_x;

	n = sizeof(x)/sizeof(x[0]);
	norm_x = 0.0;
	for (i=0; i<n; i++)	norm_x += x[i]*x[i];

	return norm_x; 
}

double norm_gpu(double *x)
{
	int n, i;
	double norm_x;

	n = sizeof(x)/sizeof(x[0]);
	norm_x = 0.0;
	#pragma acc parallel loop seq present(x)
	for (i=0; i<n; i++)	norm_x += x[i]*x[i];

	return norm_x; 
}

void initial(double *x, int N)
{
	int i;
	for (i=0; i<N; i++)	x[i] = sin(i);
}

int main()
{
	int N;
	printf(" Input N = ");
	scanf("%d", &N);

	double *x, cpu, gpu, error, t1, t2, cpu_time, gpu_time;
	x = (double *) malloc(N*sizeof(double));

	initial(x, N);
	t1 = clock();
	cpu = norm_cpu(x);
	t2 = clock();
	cpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin(x[0:N])
	{
	gpu = norm_gpu(x);
	}
	t2 = clock();
	gpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	error = fabs(cpu - gpu);
	printf(" error = %f \n", error);
	printf(" cpu times = %f , gpu times = %f \n", cpu_time, gpu_time);
	printf(" cpu/gpu = %f \n", cpu_time/gpu_time);

	return 0;
}
