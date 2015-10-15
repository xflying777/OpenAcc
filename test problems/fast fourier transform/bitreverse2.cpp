#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void cpu_bit_reverse(double *x_r, double *x_i, double *y_r, double *y_i, int N);
void gpu_bit_reverse(double *x_r, double *x_i, double *y_r, double *y_i, int N);
void Initial(double *x, double *y, int N);
int Generate_N(int p);
void error(double *y_r, double *y_i, double *z_r, double *z_i, int N);
void Print_Complex_Vector(double *y_r, double *y_i, int N);

int main()
{
	int p, N;
	double *y_r, *y_i, *z_r, *z_i, *x_r, *x_i, cpu_times, gpu_times;
	clock_t t1, t2;
	
	printf("Please input N = 2^p, p = ");
	scanf("%d",&p);
	N = Generate_N(p);
	printf("N = 2^%d = %d \n", p, N);
	
	x_r = (double *) malloc(N*sizeof(double));
	x_i = (double *) malloc(N*sizeof(double));
	y_r = (double *) malloc(N*sizeof(double));
	y_i = (double *) malloc(N*sizeof(double));
	z_r = (double *) malloc(N*sizeof(double));
	z_i = (double *) malloc(N*sizeof(double));
	
	Initial(x_r, x_i, N);
	
	t1 = clock();
	cpu_bit_reverse(x_r, x_i, y_r, y_i, N);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	gpu_bit_reverse(x_r, x_i, z_r, z_i, N);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	printf(" cpu times = %f \n gpu times = %f \n cpu times / gpu times = %f \n", cpu_times, gpu_times, cpu_times/gpu_times);
//	error(y_r, y_i, z_r, z_i, N);
	
//	Print_Complex_Vector(y_r, y_i, N);
}

void Initial(double *x, double *y, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		x[n] = n;
		y[n] = 0.0;
	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	return N;
}

void Print_Complex_Vector(double *y_r, double *y_i, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		if (y_i[n] >=0) printf("%d : %f +%f i\n", n, y_r[n], y_i[n]);
		else printf("%d : %f %f i\n", n, y_r[n], y_i[n]);
	}
}

void error(double *y_r, double *y_i, double *z_r, double *z_i, int N)
{
	int i;
	for(i=0;i<N;i++)
	{
		if(y_r[i] != z_r[i] | y_i[i] != z_i[i])
		{
			printf("error \n");
			exit(1);
		}
	}
	printf("OpenACC test was successful! \n");
		
}

void cpu_bit_reverse(double *x_r, double *x_i, double *y_r, double *y_i, int N)
{
	int k, n, i, j, M, *temp;
	double t;
	
	temp = (int *) malloc(N*sizeof(int));
	temp[0] = 0;
	
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}
	
	n = 1;
	for(M=N/2;M>0;M=M/2)
	{
		for(i=n;i<2*n;i++)
		{
			temp[i] = temp[i-n] + M; 
		}
		n = n * 2;
	}
	
//	for (i=0;i<N;i++) printf("temp[%d] = %d \n", i, temp[i]);
}

void gpu_bit_reverse(double *x_r, double *x_i, double *y_r, double *y_i, int N)
{
	int k, n, i, j, M, *temp;
	double t;
	
	temp = (int *) malloc(N*sizeof(int));
	temp[0] = 0;
	
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}
	
	#pragma acc kernels
	n = 1;
//	#pragma acc loop seq
	for(M=N/2;M>0;M=M/2)
	{
//	#pragma acc loop independent
		for(i=n;i<2*n;i++)
		{
			temp[i] = temp[i-n] + M; 
		}
		n = n * 2;
	}
	
//	for (i=0;i<N;i++) printf("temp[%d] = %d \n", i, temp[i]);
}

