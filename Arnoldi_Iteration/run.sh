
 pgc++ filename.cpp -L/usr/lib -lblas -lm

 pgc++ -acc -ta=tesla:managed -Minfo -L/usr/lib -lblas -lm filename.cpp

 pgc++ -acc -ta=tesla:managed -Minfo -Mcudalib=cublas filename.cpp

