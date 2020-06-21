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
char Arg_print_time = 0;
int Arg_mode = 0;
char* Arg_in_fname;
char* Arg_ker_fname;
float Arg_s_in;
float Arg_s_ker;
bool Arg_mem_r = false;

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

void transposeKernel0132(Tensor<float>& ker_tensor)
{
    int kh = ker_tensor.dim[0];
    int kw = ker_tensor.dim[1];
    int od = ker_tensor.dim[2];
    int ic = ker_tensor.dim[3];
    
    vector<float> val_untrans = ker_tensor.val;

    for (int c = 0; c < ic; ++c) {
        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                for (int d = 0; d < od; ++d) {
                    ker_tensor.val[
                        i * (kw * od * ic) +
                        j * (od * ic) +
                        d * ic +
                        c
                    ] = val_untrans[
                        i * (kw * od * ic) +
                        j * (od * ic) +
                        c * od +
                        d
                    ];
                }
            }
        }
    }
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

// TODO: Try change order of loops to see performance reasonable in type order.
template<typename T>
void doConv2D(
        int batch, int ih, int iw, int ic, 
        int kh, int kw, int od,
        int oh, int ow,
        Tensor<T>& padded_tensor, Tensor<T>& ker_tensor, Tensor<T>& out_tensor)
{
    clock_t start_c = clock();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    T acc = 0;
                    for (int c = 0; c < ic; ++c) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                acc += padded_tensor.val[
                                    b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic
                                    + c
                                ] * ker_tensor.val[
                                    di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic
                                    + c
                                ];
                            }
                        }
                    }
                    out_tensor.val[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    if (Arg_print_time == 'c') {
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
    
    doConv2D<float>(
            batch, ih, iw, ic, kh, kw, od, oh, ow,
            padded_tensor, ker_tensor, out_tensor);
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

    clock_t quan_c = 0;
    clock_t start_c = clock();
    Tensor<T> padded_tensor = getQuantized<T>(s_in, unquan_padded_tensor);
    Tensor<T> quan_ker_tensor = getQuantized<T>(s_ker, ker_tensor);
    quan_c += clock() - start_c;

    Tensor<T> out_tensor;
    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    doConv2D<T>(
            batch, ih, iw, ic, kh, kw, od, oh, ow,
            padded_tensor, quan_ker_tensor, out_tensor);

    start_c = clock();
    const Tensor<float>& fin_out_tensor = getDequantized(s_in * s_ker, out_tensor);
    quan_c += clock() - start_c;
    if (Arg_print_time == 'q') {
        cout << (double) quan_c / CLOCKS_PER_SEC << endl;
    }
    return fin_out_tensor;
}

bool initArgs(int argc, char* argv[]) {
    Arg_print_time = false;
    Arg_mode = 0;
    Arg_s_in = 0;
    Arg_s_ker = 0;
    Arg_mem_r = false;

    int op_c;
    while ((op_c = getopt(argc, argv, "p:ri:k:")) != -1) {
        if (op_c == 'p') {
            Arg_print_time = *optarg;
        } else if (op_c == 'r') {
            Arg_mem_r = true;
        } else if (op_c == 'i') {
            Arg_s_in = atof(optarg);
        } else if (op_c == 'k') {
            Arg_s_ker = atof(optarg);
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
    if (op_i + 2 < argc) {
        Arg_mode = atoi(argv[op_i + 2]);
        if (!(Arg_mode == 0 || 
                Arg_mode == 32 || Arg_mode == 16 || Arg_mode == 8)) {
            return false;
        }
    }

    if (Arg_mode == 32 && (Arg_s_in == 0 || Arg_s_ker == 0)) {
        Arg_s_in = -79.6f; 
        Arg_s_ker = 1022571.0f;
    } else if (Arg_mode == 16 && (Arg_s_in == 0 || Arg_s_ker == 0)) {
        Arg_s_in = -5.0f;
        Arg_s_ker = 300.7f;
    } else if (Arg_mode == 8 && (Arg_s_in == 0 || Arg_s_ker == 0)) {
        Arg_s_in = -0.22223f; 
        Arg_s_ker = 70.0f;
    }
    return true;
}

int main(int argc, char* argv[])
{
    if (!initArgs(argc, argv)) {
        cout << "Invalid args." << endl;
        return 0;
    }
    assert(Arg_mode == 0 || (Arg_s_in != 0 && Arg_s_ker != 0));

    Tensor<float> in_tensor;
    Tensor<float> ker_tensor;

    if (!readFile(Arg_in_fname, in_tensor) || !readFile(Arg_ker_fname, ker_tensor)) {
        cout << "No such file for input_tensor or kernel_tensor." << endl;
        return 0;
    }
    if (Arg_mem_r) {
        transposeKernel0132(ker_tensor);
    }

    constexpr char out_fname[] = "output_tensor.bin";
    if (Arg_mode == 0) {
        writeFile(out_fname, conv2D(in_tensor, ker_tensor));
    } else if (Arg_mode == 32) {
        writeFile(out_fname, quanConv2D<int32_t>(Arg_s_in, Arg_s_ker, in_tensor, ker_tensor));
    } else if (Arg_mode == 16) {
        writeFile(out_fname, quanConv2D<int16_t>(Arg_s_in, Arg_s_ker, in_tensor, ker_tensor));
    } else if (Arg_mode == 8) {
        writeFile(out_fname, quanConv2D<int8_t>(Arg_s_in, Arg_s_ker, in_tensor, ker_tensor));
    } else {
        assert(0);
    }
}