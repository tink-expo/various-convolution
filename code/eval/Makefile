all: nrmse

nrmse: nrmse.cpp
	g++ -std=c++11 -Wall -O3 -o $@ $^

clean:
	rm -f nrmse output_tensor.bin