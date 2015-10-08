#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	int i, n;
	double pi, t;
	n = 10000;
	pi = 0.0;
	
	#pragma acc kernels
	for(i=0;i<n;i++)
	{
		t = (i + 0.5)/n;
		pi = pi + 4.0/(1.0 + t*t);
	}
	printf("pi = %f \n", pi/n);
}
