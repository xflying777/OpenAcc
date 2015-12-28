#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void initial(float *data, int Nx, int Ny)
{
	#pragma acc loop independent
	for (int i=0;i<Ny;i++)
	{
		#pragma acc loop independent
		for (int j=0;j<Nx;j++)
		{
			data[Nx*i+j] = 1.0*j;
		}
	}	
}

void transpose(float *data, int Nx, int Ny)
{
	int i, j;
	float *temp;
	temp = (float *)malloc(Nx*Ny*sizeof(float));
	
	#pragma acc data create(temp[0:Nx*Ny]), copyin(data[0:Nx*Ny])
	#pragma acc loop independent
	for(i=0;i<Nx*Ny;i++)
	{
		temp[i] = data[i];
	}
	
	#pragma acc loop independent
	for (i=0; i<Ny; i++)
	{
		#pragma acc loop independent
		for (j=0; j<Nx; j++)
		{
			data[i+j*Ny] = temp[Nx*i+j]; 
		}
	}
	
}

void print_matrix(float *x, int Nx, int Ny)
{
	int i, j;
	for(i=0;i<Ny;i++)
	{
		for (j=0;j<Nx;j++) printf(" %f ", x[Nx*i+j]);
		printf("\n");
	}
}

int main()
{
	int Nx, Ny;
	float *a;
	
	printf(" Please input Nx = \n");
	scanf("%d",&Nx);
	printf(" Nx = %d \n", Nx);
	printf(" Please input Ny = \n");
	scanf("%d",&Ny);
	printf(" Ny = %d \n", Ny);
	
	a = (float *)malloc(Nx*Ny*sizeof(float));
	
	#pragma acc kernels copy(a[0:Nx*Ny])
	initial(a, Nx, Ny);
	printf(" Initial data[%d][%d] \n", Ny, Nx);
	print_matrix(a, Nx, Ny);
	
	#pragma acc kernels copy(a[0:Nx*Ny])
	transpose(a, Nx, Ny);
	printf(" Initial data[%d][%d] \n", Nx, Ny);
	print_matrix(a, Ny, Nx);
	
	return 0;
}


