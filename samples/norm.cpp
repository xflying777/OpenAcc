#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double norm_cpu(double *x, int N)
{
	int i;
	double norm_x;

	norm_x = 0.0;
	for (i=0; i<N; i++)	norm_x = norm_x + x[i]*x[i];

	return norm_x;
}

double norm_gpu_seq(double *x, int N)
{
	int i;
	double norm_x;

	norm_x = 0.0;
	#pragma acc parallel loop seq present(x)
	for (i=0; i<N; i++)	norm_x += x[i]*x[i];

	return norm_x;
}

double norm_gpu_reduction(double *x, int N)
{
	int i;
	double norm_x;

	#pragma acc data present(x) copyout(norm_x)
	{
		norm_x = 0.0;
		#pragma acc update device(norm_x)
		#pragma acc parallel loop reduction(+:norm_x)
		for (i=0; i<N; i++)	norm_x += x[i]*x[i];
	}
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

	double *x, cpu, gpu_seq, gpu_reduction, error, t1, t2, cpu_time, gpu_seq_time, gpu_reduction_time;
	x = (double *) malloc(N*sizeof(double));

	initial(x, N);
	printf("\n x[1] = %f \n", x[1]);

	t1 = clock();
	cpu = norm_cpu(x, N);
	t2 = clock();
	cpu_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	#pragma acc data copyin(x[0:N])
	{
		gpu_seq = norm_gpu_seq(x, N);
	}
	t2 = clock();
	gpu_seq_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	#pragma acc data copyin(x[0:N])
	{
		gpu_reduction = norm_gpu_reduction(x, N);
	}
	t2 = clock();
	gpu_reduction_time = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	error = fabs((cpu - gpu_seq) + (cpu - gpu_reduction));
	printf("\n norm = %f \n", cpu);
	printf(" error = %f \n", error);
	printf(" cpu times = %f , gpu seq times = %f , gpu reduction times = %f \n", cpu_time, gpu_seq_time, gpu_reduction_time);
	//printf(" cpu/gpu = %f \n", cpu_time/gpu_time);

	return 0;
}
