

def:
	@echo make s1 - sample s1
	@echo make s2 - sample s2
	@echo make s3 - sample s3

s0: makefile s0.cpp
	pgc++ -w -O2 -o s0 -Minfo s0.cpp

s1: makefile s1.cpp
	pgc++ -w -Kieee -O2 -o s1 -acc -Minfo=accel s1.cpp

t1: makefile t1.cpp
	pgc++ -w -Kieee -O2 -o t1 -acc -Minfo=accel t1.cpp

s2: makefile s2.cpp
	pgc++ -w -Kieee -O2 -o s2 -acc -Minfo=accel s2.cpp

s3: makefile s3.cpp
	pgc++ -w -Kieee -O2 -o s3 -acc -Minfo=accel s3.cpp

clean:
	rm -f a.out s? t? *.o
