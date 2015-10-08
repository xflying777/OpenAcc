/*
 * Simple test
 *  allocates two vectors, fills the vectors, does axpy a couple of times,
 *  compares the results for correctness.
 *  This uses a C++ templated class that acts much like std::vector,
 *  but isn't.
 */

#include <iostream>
#include <cstdio>
#include <openacc.h>

template<typename vtype>
class myvector{
    vtype* data;
    size_t size;
public:
    inline vtype & operator[]( int i ) const { return data[i]; }
    myvector( int size_ ){
	size = size_;
	data = new vtype[size];
	#pragma acc enter data create(this)
	#pragma acc enter data create(data[0:size])
    }
    ~myvector(){
	#pragma acc exit data delete(data[0:size])
	#pragma acc exit data delete(this)
	delete [] data;
    }
    void updatehost(){
	#pragma acc update host(data[0:size])
    }
    void updatedev(){
	#pragma acc update device(data[0:size])
    }
};


template<typename vtype>
static void
axpy( myvector<vtype>& y, myvector<vtype>& x, vtype a, int n ){
    #pragma acc parallel loop present(x,y)
    for( int i = 0; i < n; ++i )
        y[i] += a*x[i];
}

template<typename vtype>
static void
test( myvector<vtype>& a, myvector<vtype>& b, int n ){
    axpy<vtype>( a, b, 2.0, n );

    for( int i = 0; i < n; ++i ) b[i] = 2.0;
    b.updatedev();

    axpy<vtype>( a, b, 1.0, n );
}

/* Depending on the GCC version you have installed, this cause warnings */
extern "C" int atoi(const char*);

int main( int argc, char* argv[] ){
    int n = 1000;
    if( argc > 1 ) n = atoi(argv[1]);

    myvector<double> a(n), b(n);

    for( int i = 0; i < n; ++i ) a[i] = i;
    for( int i = 0; i < n; ++i ) b[i] = n-i;
    a.updatedev();
    b.updatedev();

    test<double>( a, b, n );
    a.updatehost();
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
