all: convolution

convolution: conv4.cu
	nvcc -Xcompiler -Wall -Xcompiler -O3 -o $@ $^

clean:
	rm -f convolution output_tensor.bin