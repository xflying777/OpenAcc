#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main()
{
	int i, n;
	double cpu_pi, gpu_pi, temp, t1, t2, gpu_times, cpu_times;
	
	printf("Input iteration n = ");
	scanf("%d",&n);

	gpu_pi = 0.0;
	t1 = clock();
	#pragma acc kernels
	for(i=0;i<n;i++)
	{
		temp = (i + 0.5)/n;
		gpu_pi = gpu_pi + 4.0/(1.0 + temp*temp);
	}	
	t2 = clock();
	gpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	cpu_pi = 0.0;
        t1 = clock();
        for(i=0;i<n;i++)
        {
                temp = (i + 0.5)/n;
                cpu_pi = cpu_pi + 4.0/(1.0 + temp*temp);
        }
        t2 = clock();
        cpu_times = 1.0*(t2 - t1)/CLOCKS_PER_SEC;


	printf(" cpu_pi = %f \n", cpu_pi/n);
	printf(" gpu_pi = %f \n", gpu_pi/n);
	printf(" cpu times = %f \n gpu times = %f \n", cpu_times, gpu_times);
	printf(" cpu times / gpu times = %f \n", cpu_times/gpu_times);
}


