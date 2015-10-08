#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	int i, n;
	double pi, t, t1, t2;
	n = 100000000;
	pi = 0.0;
	
	t1 = clock();
	#pragma acc kernels
	for(i=0;i<n;i++)
	{
		t = (i + 0.5)/n;
		pi = pi + 4.0/(1.0 + t*t);
	}
	t2 = clock();
	printf("pi = %f \n times = %f \n", pi/n, 1.0*(t2 - t1)/CLOCKS_PER_SEC);
}
