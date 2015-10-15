#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main()
{
	int i, n, m, k;
	double pi, temp, t1, t2, *t, everage;
	m = 20;
	t = (double*) malloc(m*sizeof(double));
	n = 10000000;
	
//	#pragma acc kernels
	for(k=0;k<m;k++)
	{
		pi = 0.0;
		t1 = clock();
		#pragma acc kernels
		for(i=0;i<n;i++)
		{
			temp = (i + 0.5)/n;
			pi = pi + 4.0/(1.0 + temp*temp);
		}	
		t2 = clock();
		t[k] = 1.0*(t2 - t1)/CLOCKS_PER_SEC;
	}
	everage = 0.0;
	for(i=0;i<m;i++) everage = everage + t[i];
	everage = everage / m;
	printf("pi = %f \n", pi/n);
	printf("everage times = %f \n", everage);
}


