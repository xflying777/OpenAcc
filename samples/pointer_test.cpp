#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test(double *r);

int main()
{
	double *r;
	
	r = (double *) malloc(1*sizeof(double));
	
	*r = 1.0;
	printf(" *r = %f \n", *r);
	
	*r = -1.0* *r;
	test(r);
	
	return 0;
}

void test(double *r)
{
	printf(" test *r = %f \n", *r);
}
