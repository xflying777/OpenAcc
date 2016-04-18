
 g++ filename -L/usr/lib -lblas -lfftw3 -lm

 pgc++ -acc -ta=tesla -Mcudalib=cufft,cublas -Minfo -filename.cpp

