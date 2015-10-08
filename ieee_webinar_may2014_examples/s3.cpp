/*
 * Simple test
 *  allocates two vectors, fills the vectors, does axpy a couple of times,
 *  compares the results for correctness.
 *  This is much like s2.cpp, but changes the constructor/destructor,
 *  and moves the axpy routine into the myvector class.
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
    }
    ~myvector(){
	delete [] data;
    }
    void devcopyin(){
	#pragma acc enter data create(this)
	#pragma acc enter data copyin(data[0:size])
    }
    void devcopyout(){
	#pragma acc exit data copyout(data[0:size])
	#pragma acc exit data delete(this)
    }
    void devdelete(){
	#pragma acc exit data delete(data[0:size])
	#pragma acc exit data delete(this)
    }
    void updatehost(){
	#pragma acc update host(data[0:size])
    }
    void updatedev(){
	#pragma acc update device(data[0:size])
    }
    void axpy( myvector<vtype>& x, vtype a ){
	#pragma acc parallel loop present(this,x)
	for( int i = 0; i < size; ++i )
	    data[i] += a*x[i];
    }
};

template<typename vtype>
static void
test( myvector<vtype>& a, myvector<vtype>& b, int n ){
    a.devcopyin();
    b.devcopyin();
    a.axpy( b, 2.0 );

    for( int i = 0; i < n; ++i ) b[i] = 2.0;
    b.updatedev();

    a.axpy( b, 1.0 );
    a.devcopyout();
    b.devdelete();
}

/* Depending on the GCC version you have installed, this cause warnings */
extern "C" int atoi(const char*);

int main( int argc, char* argv[] ){
    int n = 1000;
    if( argc > 1 ) n = atoi(argv[1]);

    myvector<double> a(n), b(n);

    for( int i = 0; i < n; ++i ) a[i] = i;
    for( int i = 0; i < n; ++i ) b[i] = n-i;

    test<double>( a, b, n );
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
