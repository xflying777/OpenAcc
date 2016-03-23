
 g++ filename.cpp -L/usr/lib -lblas -lm

 pgc++ -acc -ta=tesla -Minfo -L/usr/lib -lblas -lm filename.cpp

 pgc++ -acc -ta=tesla -Minfo -Mcudalib=cublas filename.cpp

