
 g++ filename.cpp -L/usr/lib -lblas -lm

 pgc++ -acc -ta=tesla:managed -Mcudalib=cublas -Minfo -L/usr/lib -lblas -lm filename.cpp

