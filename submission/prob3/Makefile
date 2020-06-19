all: convolution

convolution: conv3.cpp
	g++ -std=c++11 -Wall -O3 -o $@ $^ -mavx2 -lpthread

clean:
	rm -f convolution output_tensor.bin