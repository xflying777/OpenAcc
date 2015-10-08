/*
 * Simple test
 *  allocates two vectors, fills the vectors, does axpy a couple of times,
 *  compares the results for correctness.
 *  This example won't work with PGI 14.4; it uses the STL vector type,
 *  and needs the 'deep copy' feature that the OpenACC committee members
 *  are designing.
 */

#include <iostream>
#include <cstdio>
#include <vector>
#include <openacc.h>

template<typename vtype>
static void
axpy( std::vector<vtype>& y, std::vector<vtype>& x, vtype a ){
    #pragma acc parallel loop present(x,y)
    for( int i = 0; i < y.size(); ++i )
        y[i] += a*x[i];
}

template<typename vtype>
static void
test( std::vector<vtype>& a, std::vector<vtype>& b ){
    #pragma acc data copy(a) copyin(b)
    {
    axpy( a, b, 2.0 );

    for( int i = 0; i < b.size(); ++i ) b[i] = 2.0;
    #pragma acc update device(b)

    axpy( a, b, 1.0 );
    }
}

/* Depending on the GCC version you have installed, this cause warnings */
extern "C" int atoi(const char*);

int main( int argc, char* argv[] ){
    int n = 1000;
    if( argc > 1 ) n = atoi(argv[1]);

    std::vector<double> a(n), b(n);

    for( int i = 0; i < n; ++i ) a[i] = i;
    for( int i = 0; i < n; ++i ) b[i] = n-i;

    test<double>( a, b );
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
