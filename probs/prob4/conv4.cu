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

using namespace std;
bool arg_print_time = false;

template <typename T> 
struct Tensor {
    array<int, 4> dim;
    vector<T> val;

    inline T* valPtr()
    {
        return const_cast<T*>(val.data());
    }
};

void writeFile(const char* fname, const Tensor<float>& tensor)
{
    ofstream ofs(fname, ios::binary);
    ofs.write((const char*) (tensor.dim.data()), 16);
    ofs.write((const char*) tensor.val.data(),
            sizeof(float) * tensor.val.size());
}

bool readFile(const char* fname, Tensor<float>& tensor)
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

// in_tensor always float
template<typename T>
Tensor<T> getQuantized(float s_val, Tensor<float>& in_tensor)
{
    Tensor<T> quan_in_tensor;
    quan_in_tensor.dim = in_tensor.dim;
    quan_in_tensor.val.assign(in_tensor.val.size(), 0);
    for (size_t i = 0; i < in_tensor.val.size(); ++i) {
        quan_in_tensor.val[i] = (T) (in_tensor.val[i] * s_val);
    }
    return quan_in_tensor;
}

template<typename T>
Tensor<float> getDequantized(float s_val, Tensor<T>& quan_out_tensor)
{
    Tensor<float> fin_out_tensor;
    fin_out_tensor.dim = quan_out_tensor.dim;
    fin_out_tensor.val.assign(quan_out_tensor.val.size(), 0);
    for (size_t i = 0; i < quan_out_tensor.val.size(); ++i) {
        fin_out_tensor.val[i] = (float) quan_out_tensor.val[i] / s_val;
    }
    return fin_out_tensor;
}

template<typename T>
vector<T> getKernelTranspose(
        const Tensor<T>& ker_tensor)
{
    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    int ic = ker_tensor.dim[3];

    vector<T> trans_ker(ker_tensor.val.size());

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

template<typename T>
vector<T> getIm2col(
        int oh, int ow, int kh, int kw,
        const Tensor<T>& padded_tensor)
{
    int batch = padded_tensor.dim[0];
    int ih = padded_tensor.dim[1];
    int iw = padded_tensor.dim[2];
    int ic = padded_tensor.dim[3];
    
    int col_h = batch * oh * ow;
    int col_w = ic * kh * kw;
    vector<T> col(col_h * col_w);

    for (int i = 0; i < oh; ++i) {
        for (int j = 0; j < ow; ++j) {
            for (int c = 0; c < ic; ++c) {
                int col_i = i * ow + j;
                int col_j = c * (kh * kw);
                for (int di = 0; di < kh; ++di) {
                    for (int dj = 0; dj < kw; ++dj) {
                        col[
                            col_i * col_w +
                            col_j + (di * kw) + dj
                        ] = padded_tensor.val[
                            (i + di) * (iw * ic) +
                            (j + dj) * ic +
                            c
                        ];
                    }
                }
            }
        }
    }
    return col;
}

// TODO: Try change order of loops to see performance reasonable in type order.
template<typename T>
void doConv2D(
        Tensor<T>& padded_tensor, Tensor<T>& ker_tensor, Tensor<T>& out_tensor)
{
    clock_t start_c = clock();

    int oh = out_tensor.dim[1];
    int ow = out_tensor.dim[2];
    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    int ic = ker_tensor.dim[3];

    const vector<T>& trans_ker = getKernelTranspose(ker_tensor);
    const vector<T>& col = getIm2col(oh, ow, kh, kw, padded_tensor);

    int m_size = oh * ow;
    int n_size = od;
    int k_size = ic * kh * kw;

    for (int i = 0; i < m_size; ++i) {
        for (int j = 0; j < n_size; ++j) {
            T acc = 0;
            for (int k = 0; k < k_size; ++k) {
                acc += col[i * k_size + k] * trans_ker[k * n_size + j];
            }
            out_tensor.val[i * n_size + j] = acc;
        }
    }

    if (arg_print_time) {
        cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
    }
}

Tensor<float> getPadded(
        int ih, int pad_top,
        int iw, int pad_left,
        Tensor<float>& in_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];
    
    Tensor<float> padded_tensor;
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

Tensor<float> conv2D(Tensor<float>& in_tensor, Tensor<float>& ker_tensor)
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

    Tensor<float> out_tensor;
    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    Tensor<float> padded_tensor = getPadded(
            ih, pad_top,
            iw, pad_left,
            in_tensor);
    
    doConv2D<float>(padded_tensor, ker_tensor, out_tensor);
    return out_tensor;
}

template <typename T>
Tensor<float> quanConv2D(float s_in, float s_ker, Tensor<float>& in_tensor, Tensor<float>& ker_tensor)
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

    Tensor<float> unquan_padded_tensor = getPadded(
            ih, pad_top,
            iw, pad_left,
            in_tensor);
    Tensor<T> padded_tensor = getQuantized<T>(s_in, unquan_padded_tensor);
    Tensor<T> quan_ker_tensor = getQuantized<T>(s_ker, ker_tensor);

    Tensor<T> out_tensor;
    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    doConv2D<T>(padded_tensor, quan_ker_tensor, out_tensor);

    return getDequantized(s_in * s_ker, out_tensor);
}


int main(int argc, char* argv[])
{
    if (argc < 6) {
        cout << "Invalid args." << endl;
        return 0;
    }
    if (argc >= 7 && string(argv[6]) == "pt") {
        arg_print_time = true;
    }

    int mode = atoi(argv[3]);
    float s_in = atof(argv[4]);
    float s_ker = atof(argv[5]);

    Tensor<float> in_tensor;
    Tensor<float> ker_tensor;

    if (!readFile(argv[1], in_tensor) || !readFile(argv[2], ker_tensor)) {
        cout << "Invalid args." << endl;
        return 0;
    }

    constexpr char out_fname[] = "output_tensor.bin";
    if (mode == 0) {
        writeFile(out_fname, conv2D(in_tensor, ker_tensor));
    } else if (mode == 32) {
        writeFile(out_fname, quanConv2D<int32_t>(s_in, s_ker, in_tensor, ker_tensor));
    } else if (mode == 16) {
        writeFile(out_fname, quanConv2D<int16_t>(s_in, s_ker, in_tensor, ker_tensor));
    } else if (mode == 8) {
        writeFile(out_fname, quanConv2D<int8_t>(s_in, s_ker, in_tensor, ker_tensor));
    } else {
        cout << "Invalid args." << endl;
    }
}