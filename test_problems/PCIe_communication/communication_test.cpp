#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void communication(float *x, int n);

int main()
{
	int n, i;
	float t1, t2, *x;
	
	printf(" Please input n = ");
	scanf("%d", &n);
	
	x = (float *)malloc(n*sizeof(float));
	
	for (i = 0;i < n; i++) x[i] = 1.0*i;
	
	t1 = clock();
	communication(x, n);
	t2 = clock();
	
	printf("communication time = %f \n", 1.0*(t2-t1)/CLOCKS_PER_SEC);
	
	return 0;
}

void communication(float *x, int n)
{
	#pragma acc data copy(x[0:n])
}
