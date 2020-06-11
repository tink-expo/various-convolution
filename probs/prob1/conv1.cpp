#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <ctime>

using namespace std;

struct Tensor {
    int dim[4];
    float* val;
};

void writeFile(const string& fname, Tensor& tensor)
{
    ofstream ofs(fname, ios::binary);
    ofs.write((char*) tensor.dim, 16);
    ofs.write((char*) tensor.val, 
            sizeof(float) * tensor.dim[0] * tensor.dim[1] * tensor.dim[2] * tensor.dim[3]);
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
    
    ifs.read((char*) tensor.dim, 16);
    tensor.val = (float*) calloc(fsize - 16, sizeof(float));
    ifs.read((char*) tensor.val, fsize - 16);
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
    *pad_back = *out_size - *pad_front;
}

void doPad(int pad_top, int pad_bottom, 
        int pad_left, int pad_right, 
        Tensor& in_tensor)
{
    int batch = in_tensor.dim[0];
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];

    int ih = np_ih + pad_top + pad_bottom;
    int iw = np_iw + pad_left + pad_right;

    in_tensor.dim[1] = ih;
    in_tensor.dim[2] = iw;

    float* val_padded = (float*) calloc(batch * ih * iw * ic, sizeof(float));

    float (*val_padded_arr)[ih][iw][ic] = (float (*)[ih][iw][ic]) val_padded;
    float (*val_arr)[np_ih][np_iw][ic] = (float (*)[np_ih][np_iw][ic]) in_tensor.val;

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < np_ih; ++i) {
            for (int j = 0; j < np_iw; ++j) {
                for (int c = 0; c < ic; ++c) {
                    val_padded_arr[b][i + pad_top][j + pad_left][c] =
                            val_arr[b][i][j][c];
                }
            }
        }
    }
    free(in_tensor.val);
    in_tensor.val = val_padded;
}

clock_t conv2D(Tensor& in_tensor, Tensor& ker_tensor, Tensor& out_tensor)
{
    int np_ih = in_tensor.dim[1];
    int np_iw = in_tensor.dim[2];

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

    doPad(pad_top, pad_bottom,
            pad_left, pad_right,
            in_tensor);

    int batch = in_tensor.dim[0];
    int ih = in_tensor.dim[1];
    int iw = in_tensor.dim[2];
    int ic = in_tensor.dim[3];

    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val = (float*) calloc(batch * oh * ow * od, sizeof(float));

    float (*in_arr)[ih][iw][ic] = (float (*)[ih][iw][ic])(in_tensor.val);
    float (*ker_arr)[kw][od][ic] = (float (*)[kw][od][ic])(ker_tensor.val);
    float (*out_arr)[oh][ow][od] = (float (*)[oh][ow][od])(out_tensor.val);

    clock_t start_c = clock();
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int c = 0; c < ic; ++c) {
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                out_arr[b][i][j][d] +=
                                        in_arr[b][i + di][j + dj][c] * 
                                        ker_arr[di][dj][d][c];
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
    writeFile("out_tensor.bin", out_tensor);

    free(in_tensor.val);
    free(ker_tensor.val);
    free(out_tensor.val);
}