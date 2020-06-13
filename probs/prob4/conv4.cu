#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <ctime>
#include <array>
#include <limits>
#include <unistd.h>

using namespace std;

// Global args.
bool Arg_print_time = false;
char* Arg_in_fname;
char* Arg_ker_fname;

constexpr int CUDA_THREADS_2D = 16;

struct Tensor {
    array<int, 4> dim;
    vector<float> val;

    inline float* valPtr()
    {
        return const_cast<float*>(val.data());
    }
};

void writeFile(const char* fname, const Tensor& tensor)
{
    ofstream ofs(fname, ios::binary);
    ofs.write((const char*) (tensor.dim.data()), 16);
    ofs.write((const char*) tensor.val.data(),
            sizeof(float) * tensor.val.size());
}

bool readFile(const char* fname, Tensor& tensor)
{
    ifstream ifs(fname, ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    ifs.seekg(0, ios::end);
    size_t fsize = ifs.tellg();
    ifs.seekg(0, ios::beg);
    
    ifs.read((char*) const_cast<int*>(tensor.dim.data()), 16);
    tensor.val.assign((fsize - 16) / sizeof(float), 0);
    ifs.read((char*) tensor.valPtr(), fsize - 16);
    return true;
}

void getOutPads1D(int in_size, int ker_size, int* out_size, int* pad_front, int* pad_back)
{
    int stride_size = 1;
    *out_size = static_cast<int>(ceil(static_cast<double>(in_size) / stride_size));
    int pad_size = max(
            (*out_size - 1) * stride_size + ker_size - in_size,
            0);
    *pad_front = pad_size / 2;
    *pad_back = pad_size - *pad_front;
}

vector<float> getKernelTranspose(
        const Tensor& ker_tensor)
{
    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    int ic = ker_tensor.dim[3];

    vector<float> trans_ker(ker_tensor.val.size());

    for (int c = 0; c < ic; ++c) {
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                for (int d = 0; d < od; ++d) {
                    trans_ker[
                        c * (kh * kw * od) +
                        i * (kw * od) +
                        j * od +
                        d
                    ] = ker_tensor.val[
                        i * (kw * od * ic) +
                        j * (od * ic) +
                        d * ic +
                        c
                    ];
                }
            }
        }
    }
    return trans_ker;
}

vector<float> getIm2col(
        int oh, int ow, int kh, int kw,
        const Tensor& padded_tensor)
{
    int batch = padded_tensor.dim[0];
    int ih = padded_tensor.dim[1];
    int iw = padded_tensor.dim[2];
    int ic = padded_tensor.dim[3];
    
    int col_h = batch * oh * ow;
    int col_w = ic * kh * kw;
    vector<float> col(col_h * col_w);
    for (int b = 0;  b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int c = 0; c < ic; ++c) {
                    int col_i = i * ow + j;
                    int col_j = c * (kh * kw);
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            col[
                                b * (oh * ow * col_w) +
                                col_i * col_w +
                                col_j + (di * kw) + dj
                            ] = padded_tensor.val[
                                b * (ih * iw * ic) +
                                (i + di) * (iw * ic) +
                                (j + dj) * ic +
                                c
                            ];
                        }
                    }
                }
            }
        }
    }
    return col;
}

template<typename T>
__global__ void h_cuda_matmul(T* imcol, T* kernel, T* result, 
    int m_size, int n_size, int k_size)
{
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx < m_size && j_idx < n_size) {
        T acc = 0;
        for (int k = 0; k < k_size; ++k) {
            acc += imcol[i_idx * k_size + k] * kernel[k * n_size + j_idx];
        }
        result[i_idx * n_size + j_idx] = acc;
    }
}

// TODO: Try batch > 1
void conv2Dcuda(
        Tensor& padded_tensor, Tensor& ker_tensor, Tensor& out_tensor)
{
    clock_t start_c = clock();

    int batch = out_tensor.dim[0];
    int oh = out_tensor.dim[1];
    int ow = out_tensor.dim[2];

    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    int ic = ker_tensor.dim[3];

    const vector<float>& col = getIm2col(oh, ow, kh, kw, padded_tensor);
    const vector<float>& trans_ker = getKernelTranspose(ker_tensor);

    int m_size = batch * oh * ow;
    int n_size = od;
    int k_size = ic * kh * kw;

    float* d_col;
    float* d_ker;
    float* d_out;
    cudaMalloc((void **) &d_col, sizeof(float) * m_size * k_size);
    cudaMalloc((void **) &d_ker, sizeof(float) * k_size * n_size);
    cudaMalloc((void **) &d_out, sizeof(float) * m_size * k_size);

    cudaMemcpy(d_col, col.data(), sizeof(float) * m_size * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, trans_ker.data(), sizeof(float) * k_size * n_size, cudaMemcpyHostToDevice);
    
    unsigned int grid_r = (m_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
    unsigned int grid_c = (n_size + CUDA_THREADS_2D - 1) / CUDA_THREADS_2D;
    dim3 grid_dim(grid_c, grid_r);
    dim3 block_dim(CUDA_THREADS_2D, CUDA_THREADS_2D);

    h_cuda_matmul<<<grid_dim, block_dim>>>(d_col, d_ker, d_out, m_size, n_size, k_size);
    cudaFree(d_col);
    cudaFree(d_ker);

    cudaMemcpy(const_cast<float*>(out_tensor.val.data()), d_out, sizeof(float) * m_size * n_size, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    if (Arg_print_time) {
        cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
    }
}

Tensor getPadded(
        int ih, int pad_top,
        int iw, int pad_left,
        Tensor& in_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];
    
    Tensor padded_tensor;
    padded_tensor.dim = in_tensor.dim;
    padded_tensor.dim[1] = ih;
    padded_tensor.dim[2] = iw;
    padded_tensor.val.assign(batch * ih * iw * ic, 0);

    float (*padded_val_arr)[ih][iw][ic] = (float (*)[ih][iw][ic]) padded_tensor.valPtr();
    float (*val_arr)[np_ih][np_iw][ic] = (float (*)[np_ih][np_iw][ic]) in_tensor.valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < np_ih; ++i) {
            for (int j = 0; j < np_iw; ++j) {
                for (int c = 0; c < ic; ++c) {
                    padded_val_arr[b][i + pad_top][j + pad_left][c] =
                            val_arr[b][i][j][c];
                }
            }
        }
    }
    return padded_tensor;
}

Tensor conv2D(Tensor& in_tensor, Tensor& ker_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];

    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    
    int oh;
    int pad_top;
    int pad_bottom;
    getOutPads1D(np_ih, kh, &oh, &pad_top, &pad_bottom);

    int ow;
    int pad_left;
    int pad_right;
    getOutPads1D(np_iw, kw, &ow, &pad_left, &pad_right);

    int ih = np_ih + pad_top + pad_bottom;
    int iw = np_iw + pad_left + pad_right;

    Tensor out_tensor;
    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    Tensor padded_tensor = getPadded(
            ih, pad_top,
            iw, pad_left,
            in_tensor);
    
    conv2Dcuda(padded_tensor, ker_tensor, out_tensor);
    return out_tensor;
}

bool initArgs(int argc, char* argv[]) {
    Arg_print_time = false;

    int op_c;
    while ((op_c = getopt(argc, argv, "p")) != -1) {
        if (op_c == 'p') {
            Arg_print_time = true;
        } else {
            return false;
        }
    }

    int op_i = optind;
    if (op_i + 1 >= argc) {
        return false;
    }
    Arg_in_fname = argv[op_i];
    Arg_ker_fname = argv[op_i + 1];
    return true;
}

int main(int argc, char* argv[])
{
    if (!initArgs(argc, argv)) {
        cout << "Invalid args." << endl;
        return 0;
    }

    Tensor in_tensor;
    Tensor ker_tensor;

    if (!readFile(Arg_in_fname, in_tensor) || !readFile(Arg_ker_fname, ker_tensor)) {
        cout << "No such file for input_tensor or kernel_tensor." << endl;
        return 0;
    }

    constexpr char out_fname[] = "output_tensor.bin";
    writeFile(out_fname, conv2D(in_tensor, ker_tensor));
}