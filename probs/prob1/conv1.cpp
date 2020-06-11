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

using namespace std;

struct Tensor {
    array<int, 4> dim;
    vector<float> val;

    inline float* valPtr()
    {
        return const_cast<float*>(val.data());
    }
};

void writeFile(const string& fname, Tensor& tensor)
{
    ofstream ofs(fname, ios::binary);
    ofs.write((char*) const_cast<int*>(tensor.dim.data()), 16);
    ofs.write((char*) tensor.valPtr(),
            sizeof(float) * tensor.val.size());
}

bool readFile(const string& fname, Tensor& tensor)
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

Tensor getPadded(
        int ih, int pad_top,
        int iw, int pad_left,
        Tensor& in_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];

    Tensor padded_in_tensor;
    padded_in_tensor.dim = in_tensor.dim;
    padded_in_tensor.dim[1] = ih;
    padded_in_tensor.dim[2] = iw;
    padded_in_tensor.val.assign(batch * ih * iw * ic, 0);

    float (*padded_val_arr)[ih][iw][ic] = (float (*)[ih][iw][ic]) padded_in_tensor.valPtr();
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

    return padded_in_tensor;
}

clock_t conv2D(Tensor& in_tensor, Tensor& ker_tensor, Tensor& out_tensor)
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
    Tensor padded_in_tensor = getPadded(
            ih, pad_top,
            iw, pad_left,
            in_tensor);

    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    float (*padded_in_arr)[ih][iw][ic] = (float (*)[ih][iw][ic]) padded_in_tensor.valPtr();
    float (*ker_arr)[kw][od][ic] = (float (*)[kw][od][ic])ker_tensor.valPtr();
    float (*out_arr)[oh][ow][od] = (float (*)[oh][ow][od])out_tensor.valPtr();

    clock_t start_c = clock();
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int c = 0; c < ic; ++c) {
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                out_arr[b][i][j][d] +=
                                        padded_in_arr[b][i + di][j + dj][c]
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


int main(int argc, char* argv[])
{
    Tensor in_tensor;
    Tensor ker_tensor;
    Tensor out_tensor;

    if (!readFile(string(argv[1]), in_tensor) || !readFile(string(argv[2]), ker_tensor)) {
        cout << "File open failed." << endl;
        return 0;
    }

    clock_t elasped = conv2D(in_tensor, ker_tensor, out_tensor);
    cout << (double) elasped / CLOCKS_PER_SEC << endl;
    writeFile("output_tensor.bin", out_tensor);
}