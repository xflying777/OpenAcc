#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void cpuFFTr2(float *x_r, float *x_i, float *y_r, float *y_i, int N);
void gpuFFTr2(float *x_r, float *x_i, float *y_r, float *y_i, int N);
void error(float *y_r, float *y_i, float *z_r, float *z_i, int N);
void Initial(float *x, float *y, int N);
void Print_Complex_Vector(float *y_r, float *y_i, int N);
int Generate_N(int p);

int main()
{
	int p, N;
	float *y_r, *y_i, *z_r, *z_i, *x_r, *x_i, cpu_times, gpu_times;
	clock_t t1, t2;
	
	printf("Please input N = 2^p, p = ");
	scanf("%d",&p);
	N = Generate_N(p);
	printf("N = 2^%d = %d \n", p, N);
	
	x_r = (float *) malloc(N*sizeof(float));
	x_i = (float *) malloc(N*sizeof(float));
	y_r = (float *) malloc(N*sizeof(float));
	y_i = (float *) malloc(N*sizeof(float));
	z_r = (float *) malloc(N*sizeof(float));
	z_i = (float *) malloc(N*sizeof(float));
	
	Initial(x_r, x_i, N);
	
	t1 = clock();
	cpuFFTr2(x_r, x_i, y_r, y_i, N);
	t2 = clock();
	cpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;
	
	t1 = clock();
	gpuFFTr2(x_r, x_i, z_r, z_i, N);
	t2 = clock();
	gpu_times = 1.0*(t2-t1)/CLOCKS_PER_SEC;	
	
	printf("cpu times = %f \n gpu_times = %f \n cpu times / gpu times = %f \n", cpu_times, gpu_times, cpu_times/gpu_times);
	error(y_r, y_i, z_r, z_i, N);
//	Print_Complex_Vector(y_r, y_i, N);
	
	return 0;
} 

void Initial(float *x, float *y, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		x[n] = n;
		y[n] = 0.0;
	}
}

void Print_Complex_Vector(float *y_r, float *y_i, int N)
{
	int n;
	for(n=0;n<N;++n)
	{
		if (y_i[n] >=0) printf("%d : %f +%f i\n", n, y_r[n], y_i[n]);
		else printf("%d : %f %f i\n", n, y_r[n], y_i[n]);
	}
}

int Generate_N(int p)
{
	int N = 1;
	for(;p>0;p--) N*=2;
	return N;
}

void error(float *y_r, float *y_i, float *z_r, float *z_i, int N)
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

void cpuFFTr2(float *x_r, float *x_i, float *y_r, float *y_i, int N)
{
	// input : x = x_r + i * x_i
	// output: y = y_r + i * y_i
	int k, n, i, j, M;
	float t_r, t_i;
	
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}
	
	i = j = 0;
	while(i < N)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
			y_r[i] = y_r[j];
			y_r[j] = t_r;
		}
		M = N/2;
		while(j >= M & M > 0)
		{
			j = j - M;
			M = M / 2;
		}
		j = j + M;		
		i = i + 1;
	}
	// Butterfly structure
	float theta, w_r, w_i;
	n = 2;
	while(n <= N)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<N;i+=n)
			{
				j = i + n/2;
				t_r = w_r * y_r[j] - w_i * y_i[j];
				t_i = w_r * y_i[j] + w_i * y_r[j];
				

				y_r[j] = y_r[i] - t_r;
				y_i[j] = y_i[i] - t_i;
				y_r[i] = y_r[i] + t_r;
				y_i[i] = y_i[i] + t_i;

			}
		}
		n = n * 2;
	}
	
}


void gpuFFTr2(float *x_r, float *x_i, float *restrict y_r, float *restrict y_i, int N)
{
	// input : x = x_r + i * x_i
	// output: y = y_r + i * y_i
	int k, n, i, j, M;
	float t_r, t_i;
	
	#pragma acc data copyin(x_r[0:N], x_i[0:N]) copy(y_r[0:N], y_i[0:N])
	#pragma acc parallel loop independent
	for(n=0;n<N;++n)
	{
		y_r[n] = x_r[n];
		y_i[n] = x_i[n];
	}

	#pragma acc kernels
	i = j = 0;
	while(i < N)
	{
		if(i < j)
		{
			// swap y[i], y[j]
			t_r = y_r[i];
			y_r[i] = y_r[j];
			y_r[j] = t_r;
		}
		M = N/2;
		while(j >= M & M > 0)
		{
			j = j - M;
			M = M / 2;
		}
		j = j + M;		
		i = i + 1;
	}
	
	#pragma acc kernels
	// Butterfly structure
	float theta, w_r, w_i;
	n = 2;
	while(n <= N)
	{
		for(k=0;k<n/2;k++)
		{
			theta = -2.0*k*M_PI/n;
			w_r = cos(theta);
			w_i = sin(theta);
			for(i=k;i<N;i+=n)
			{
				j = i + n/2;
				t_r = w_r * y_r[j] - w_i * y_i[j];
				t_i = w_r * y_i[j] + w_i * y_r[j];
				

				y_r[j] = y_r[i] - t_r;
				y_i[j] = y_i[i] - t_i;
				y_r[i] = y_r[i] + t_r;
				y_i[i] = y_i[i] + t_i;

			}
		}
		n = n * 2;
	}
}

