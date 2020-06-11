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

template <typename T> 
struct Tensor {
    array<int, 4> dim;
    vector<T> val;

    inline T* valPtr()
    {
        return const_cast<T*>(val.data());
    }
};

// Only float.
template <typename T>
void writeFile(const string& fname, Tensor<T>& tensor)
{
    ofstream ofs(fname, ios::binary);
    ofs.write((char*) const_cast<int*>(tensor.dim.data()), 16);
    ofs.write((char*) tensor.valPtr(),
            sizeof(float) * tensor.val.size());
}

bool readFile(const string& fname, Tensor<float>& tensor)
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

template<typename T>
float getMaxS()
{
    if (sizeof(T) == sizeof(int32_t)) {
        return (float) 1e+9;
    } else if (sizeof(T) == sizeof(int16_t)) {
        return (float) 1e+4;
    } else if (sizeof(T) == sizeof(int8_t)) {
        return (float) 1e+2;
    }
    
    assert(0);
}

float getMaxAbs(const Tensor<float>& tensor)
{
    float ret = 0.0f;
    for (float v : tensor.val) {
        ret = max(ret, abs(v));
    }
    return ret;
}

template<typename T>
T getS(const Tensor<float>& in_tensor, const Tensor<float>& ker_tensor)
{
    float in_max = getMaxAbs(in_tensor);
    float ker_max = getMaxAbs(ker_tensor);
    // kh * kw * ic
    cout << in_max << " " << ker_max << endl;
    cout << ker_tensor.dim[0] << " " << ker_tensor.dim[1] << " " << ker_tensor.dim[3] << endl;
    float mult_max = in_max * ker_max * ker_tensor.dim[0] * ker_tensor.dim[1] * ker_tensor.dim[3];

    float s_max = getMaxS<T>();
    float t_max = numeric_limits<T>::max();
    cout << s_max << endl;
    if (in_max > 1.0f) {
        cout << in_max << " " << s_max << endl;
        s_max = min(s_max, t_max / in_max);
    }
    if (ker_max > 1.0f) {
        cout << ker_max << " " << s_max << endl;
        s_max = min(s_max, t_max / ker_max);
    }
    if (mult_max > 1.0f) {
        cout << mult_max << " " << s_max << endl;
        s_max = min(s_max, sqrt(t_max / mult_max));
    }
    cout << (int) (T) s_max << endl;
    return (T) s_max;
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
void getQuantized(int precision, Tensor<T>& in_tensor, Tensor<T>& quan_in_tensor)
{
    float min_val = (float) numeric_limits<T>::min();
    float max_val = (float) numeric_limits<T>::max();
    
    int s_val = (int) pow(10, precision);
    
    quan_in_tensor.dim = in_tensor.dim;
    quan_in_tensor.val.assign(in_tensor.val.size(), 0);
    for (size_t i = 0; i < in_tensor.val.size(); ++i) {
        quan_in_tensor.val[i] = (T) min(max_val, max(min_val, (float) in_tensor.val[i] * s_val));
    }
}

template<typename T>
void getDequantized(int precision, Tensor<T>& quan_out_tensor, Tensor<T>& fin_out_tensor)
{
    float s_val_sq = (float) pow(10, precision);
    s_val_sq = s_val_sq * s_val_sq;
    
    fin_out_tensor.dim = quan_out_tensor.dim;
    fin_out_tensor.val.assign(quan_out_tensor.val.size(), 0);
    for (size_t i = 0; i < quan_out_tensor.val.size(); ++i) {
        fin_out_tensor.val[i] = (float) quan_out_tensor.val[i] / s_val_sq;
    }
}

// Only float.
template <typename T> 
void getPadded(
        int ih, int pad_top,
        int iw, int pad_left,
        Tensor<float>& in_tensor,
        Tensor<T>& padded_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];
    
    padded_tensor.dim = in_tensor.dim;
    padded_tensor.dim[1] = ih;
    padded_tensor.dim[2] = iw;
    padded_tensor.val.assign(batch * ih * iw * ic, 0);

    T (*padded_val_arr)[ih][iw][ic] = (T (*)[ih][iw][ic]) padded_tensor.valPtr();
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
}

template <typename T>
clock_t conv2D(int precision, Tensor<float>& in_tensor, Tensor<float>& ker_tensor, Tensor<T>& out_tensor)
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

    Tensor<T> padded_tensor;
    if (precision) {
        Tensor<T> unquan_padded_tensor;
        getPadded<T>(
                ih, pad_top,
                iw, pad_left,
                in_tensor,
                unquan_padded_tensor);
        getQuantized<T>(precision, unquan_padded_tensor, padded_tensor);
        
    } else {
        getPadded<T>(
                ih, pad_top,
                iw, pad_left,
                in_tensor,
                padded_tensor);
    }

    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    T (*padded_arr)[ih][iw][ic] = (T (*)[ih][iw][ic]) padded_tensor.valPtr();
    T (*ker_arr)[kw][od][ic] = (T (*)[kw][od][ic])ker_tensor.valPtr();
    T (*out_arr)[oh][ow][od] = (T (*)[oh][ow][od])out_tensor.valPtr();

    clock_t start_c = clock();
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    for (int c = 0; c < ic; ++c) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                out_arr[b][i][j][d] +=
                                        padded_arr[b][i + di][j + dj][c]
                                        * ker_arr[di][dj][d][c];
                            }
                        }
                    }
                }
            }
        }
    }
    clock_t elasped_c = clock() - start_c;
    return elasped_c;
}

template<typename T>
bool conv2DWrapper(int precision, const string& in_fname, const string& ker_fname)
{
    Tensor<float> in_tensor;
    Tensor<float> ker_tensor;
    Tensor<T> out_tensor;

    if (!readFile(in_fname, in_tensor) || !readFile(ker_fname, ker_tensor)) {
        return false;
    }

    clock_t elasped = conv2D<T>(precision, in_tensor, ker_tensor, out_tensor);
    cout << (double) elasped / CLOCKS_PER_SEC << endl;

    constexpr char out_fname[] = "output_tensor.bin";
    if (precision) {
        Tensor<T> fin_out_tensor;
        getDequantized(precision, out_tensor, fin_out_tensor);
        writeFile<T>(out_fname, fin_out_tensor);
    } else {
        writeFile<T>(out_fname, out_tensor);
    }
    return true;
}


int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Invalid args." << endl;
        return 0;
    }
    int mode = argc >= 4 ? atoi(argv[3]) : 0;

    bool success = false;
    if (mode == 0) {
        success = conv2DWrapper<float>(0, argv[1], argv[2]);
    } else if (mode == 32) {
        success = conv2DWrapper<int32_t>(4, argv[1], argv[2]);
    } else if (mode == 16) {
        success = conv2DWrapper<int16_t>(3, argv[1], argv[2]);
    } else if (mode == 8) {
        success = conv2DWrapper<int8_t>(2, argv[1], argv[2]);
    }

    if (!success) {
        cout << "Invalid args." << endl;
        return 0;
    }
}