
 g++ filename.cpp -L/usr/lib -lblas -lm

 pgc++ -L/usr/lib -lblas -lm filename.cpp

 pgc++ -acc -ta=tesla:managed -Mcudalib=cufft,cublas -Minfo filname.cpp

