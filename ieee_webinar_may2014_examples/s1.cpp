/*
 * Simple test
 *  allocates two vectors, fills the vectors, does axpy a couple of times,
 *  compares the results for correctness.
 *  This is written in the C-subset of C++.
 */

#include <iostream>
#include <cstdio>
#include <openacc.h>

/* Depending on the GCC version you have installed, this cause warnings */
extern "C" int atoi(const char*);

static void
axpy( double* y, double* x, double a, int n ){
    #pragma acc parallel loop present(x[0:n],y[0:n])
    for( int i = 0; i < n; ++i )
        y[i] += a*x[i];
}

static void
test( double* a, double* b, int n ){
    #pragma acc data copy(a[0:n]) copyin(b[0:n])
    {
    axpy( a, b, 2.0, n );

    for( int i = 0; i < n; ++i ) b[i] = 2.0;
    #pragma acc update device(b[0:n])

    axpy( a, b, 1.0, n );
    }
}


int main( int argc, char* argv[] ){
    int n = 1000;
    if( argc > 1 ) n = atoi(argv[1]);

    double* a, *b;

    a = new double[n];
    b = new double[n];

    for( int i = 0; i < n; ++i ) a[i] = i;
    for( int i = 0; i < n; ++i ) b[i] = n-i;

    test( a, b, n );

    double sum = 0.0;
    for( int i = 0; i < n; ++i ) sum += a[i];
    double exp = 0.0;
    for( int i = 0; i < n; ++i ) exp += (double)i + 2.0*(double)(n-i) + 2.0;
    std::cout << "Checksum is " << sum << std::endl;
    if( exp != sum )
	std::cout << "Difference is " << exp - sum << std::endl;
    else
	std::cout << "PASSED" << std::endl;
    return 0;
}
