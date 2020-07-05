all: conv_vanila conv_cpu conv_gpu

conv_vanila: conv_vanila.cc
	g++ -std=c++11 -Wall -O3 -o conv_vanila.out $^

conv_cpu: conv_cpu.cc
	g++ -std=c++11 -Wall -O3 -o conv_cpu.out $^ -mavx2 -lpthread

conv_gpu: conv_gpu.cc
	nvcc -Xcompiler -Wall -Xcompiler -O3 -o conv_gpu.out $^

clean:
	rm -f *.out output_tensor.bin