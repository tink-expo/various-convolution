# Various Optimization Techniques for Convolution Operation

## About
This is an implementation of convolution operation using following optimization strategies:
- **Naive quantization.**
Linear quantization using uniform scale value.
- **CPU parallelization.**
Parallelization using multithreading (pthread) and AVX instructions.
- **GPU parallelization.**
Parallelization using CUDA.

## Programs
This project consists of several programs listed below. They have some duplicated code which can be shared, but they are intentionally unshared to make each programs completely independent.
### conv_vanila
Optionally uses naive quantization with scale value found by [AVM search](http://avmframework.org/) on several typical examples of input and kernel tensor files. Convolution operation is done by simple arithmetic calculation. Note that strictly speaking, this 'convolution operation' is actually 'correlation', like many machine learning frameworks' convolution operations.
### conv_cpu
Like `conv_vanila`, optionally uses naive quantization. Additionally, this program uses multithreading (pthread library), and AVX instructions for optimization.
### conv_gpu
This program uses CUDA for optimization. Unlike two programs above that uses CPU for demanding operation for convolution, this program first process input and kernel tensor using `im2col`, and performs convolution by matrix multiplication on GPU.
### nrmse
This program is for measuring quantization error. It uses normalized root-mean-square error (NRMSE).

## Usage
### Environment
Tested in Ubuntu 16.04, gcc 5.4.0, CUDA 10.1
### Input / Output file format
`conv_*` programs take 2 binary files as input.  
- **input tensor.** First 16 bytes are for `(N, H, W, IC)`, where `N` is the batch size, `H` is the height, `W` is the width, and `IC` is the channel.
- **kernel tensor.** First 16 bytes are for `(KH, KW, OC, IC)`, where `KH` is the kernel height, `KW` is the kernel width, `OC` is the output channel, and `IC` is the input channel.  

They produce 1 binary file as output.
- **output tensor.** First 16 bytes are for `(N, H, W, OC)`, where `N` is the batch size, `H` is the height, `W` is the width, and `OC` is the channel.

For all binary files, following bytes after first 16 bytes are the real tensor data, which follows the memory order corresponding to the dimension rule written above.

### Running command
At `src/` directory,
```
$ make
```

**conv_vanila**
```
$ ./conv_vanila $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH) [32/16/8]
```
With no `[32/16/8]` argument specified, no quantization is applied. Otherwise, quantization using integer of corresponding number of bits is applied.

**conv_cpu**
```
$ ./conv_cpu $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH) {FP32/INT32/INT16}
```
Third argument is mandatory. For `FP32`, no quantization is applied. For `INT*`, quantization using integer of corresponding number of bits is applied.

**conv_gpu**
```
$ ./conv_gpu $(INPUT_BIN_PATH) $(OUTPUT_BIN_PATH)
```
Quantization is not implemented for GPU version.

**nrmse**  
At `src/` directory,
```
$ make nrmse
$ ./nrmse $(X_BIN_PATH) $(Y_BIN_PATH)
```
Note that `$ make all` doesn't build `nrmse`.
