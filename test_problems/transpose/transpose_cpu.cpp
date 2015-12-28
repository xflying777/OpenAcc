#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void initial(float *data, int Nx, int Ny)
{
	for (int i=0;i<Ny;i++)
	{
		for (int j=0;j<Nx;j++)
		{
			data[Nx*i+j] = 1.0*j;
		}
	}	
}

void transpose(float *data_in, float *data_out, int Nx, int Ny)
{
	int i, j;
	for (i=0; i<Ny; i++)
	{
		for (j=0; j<Nx; j++)
		{
			data_out[i+j*Ny] = data_in[Nx*i+j]; 
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
	float *a, *b, t1, t2, t3, t4;
	
	printf(" Please input Nx = \n");
	scanf("%d",&Nx);
	printf(" Nx = %d \n", Nx);
	printf(" Please input Ny = \n");
	scanf("%d",&Ny);
	printf(" Ny = %d \n", Ny);
	
	a = (float *)malloc(Nx*Ny*sizeof(float));
	b = (float *)malloc(Nx*Ny*sizeof(float));
	
	t1 = clock();
	initial(a, Nx, Ny);
	t2 = clock();

//	printf(" Initial data[%d][%d] \n", Ny, Nx);
//	print_matrix(a, Nx, Ny);
	
	t3 = clock();
	transpose(a, b, Nx, Ny);
	t4 = clock();

//	printf(" Initial data[%d][%d] \n", Nx, Ny);
//	print_matrix(a, Ny, Nx);
	
	printf(" Initialization time = %f \n",1.0*(t2-t1)/CLOCKS_PER_SEC);
	printf(" Transpose time = %f \n",1.0*(t4-t3)/CLOCKS_PER_SEC);
	printf(" Total time = %f \n",1.0*(t4-t1)/CLOCKS_PER_SEC);

	return 0;
}


