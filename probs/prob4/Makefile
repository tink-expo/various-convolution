all: convolution

convolution: conv4.cu
	nvcc -std=c++11 -Xcompiler -Wall -O3 -o $@ $^

clean:
	rm -f convolution output_tensor.bin