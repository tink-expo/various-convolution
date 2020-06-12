all: convolution

convolution: conv4.cu
	nvcc -std=c++11 -Xcompiler -Wall -O3 -o $@ $^ -Xcompiler -mavx2

clean:
	rm -f convolution output_tensor.bin