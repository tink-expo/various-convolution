all: convolution

convolution: conv2.cpp
	g++ -std=c++11 -Wall -O3 -o $@ $^

clean:
	rm -f convolution output_tensor.bin