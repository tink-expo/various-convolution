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
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>

using namespace std;

constexpr int P_THREADS = 4;

// Global args.
bool Arg_print_time = false;
int Arg_mode = 0;
char* Arg_in_fname;
char* Arg_ker_fname;
float Arg_s_in;
float Arg_s_ker;

template <typename T> 
struct Tensor {
    array<int, 4> dim;
    vector<T> val;

    inline T* valPtr()
    {
        return const_cast<T*>(val.data());
    }
};

template <typename T>
struct ThreadArg {
    Tensor<T>* padded_tensor;
    Tensor<T>* ker_tensor;
    Tensor<T>* out_tensor;
    
    int oh_s;
    int oh_e;
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


// [DoConv2D] - No pthread. Unused.

void doConv2D(
        int batch, int ih, int iw, int ic, 
        int kh, int kw, int od,
        int oh, int ow,
        Tensor<int16_t>& padded_tensor, Tensor<int16_t>& ker_tensor, Tensor<int16_t>& out_tensor)
{
    clock_t start_c = clock();
    int16_t* padded_val_ptr = (int16_t*) padded_tensor.valPtr();
    int16_t* ker_val_ptr = (int16_t*) ker_tensor.valPtr();
    int16_t* out_val_ptr = (int16_t*) out_tensor.valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int16_t acc = 0;
                    __m256i r_av = _mm256_setzero_si256();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 16; c += 16) {
                                __m256i in_av = _mm256_loadu_si256((__m256i*) (padded_val_ptr + i_idx + c));
                                __m256i k_av = _mm256_loadu_si256((__m256i*) (ker_val_ptr + k_idx + c));
                                __m256i mu_av = _mm256_mullo_epi16(in_av, k_av);
                                r_av = _mm256_adds_epi16(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    int16_t* r_av_ptr = (int16_t*)&r_av;
                    for (int avi = 0; avi < 16; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    if (Arg_print_time) {
        cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
    }
}

void doConv2D(
        int batch, int ih, int iw, int ic, 
        int kh, int kw, int od,
        int oh, int ow,
        Tensor<int32_t>& padded_tensor, Tensor<int32_t>& ker_tensor, Tensor<int32_t>& out_tensor)
{
    clock_t start_c = clock();
    int32_t* padded_val_ptr = (int32_t*) padded_tensor.valPtr();
    int32_t* ker_val_ptr = (int32_t*) ker_tensor.valPtr();
    int32_t* out_val_ptr = (int32_t*) out_tensor.valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int32_t acc = 0;
                    __m256i r_av = _mm256_setzero_si256();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 8; c += 8) {
                                __m256i in_av = _mm256_loadu_si256((__m256i*) (padded_val_ptr + i_idx + c));
                                __m256i k_av = _mm256_loadu_si256((__m256i*) (ker_val_ptr + k_idx + c));
                                __m256i mu_av = _mm256_mullo_epi32(in_av, k_av);
                                r_av = _mm256_add_epi32(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    int32_t* r_av_ptr = (int32_t*)&r_av;
                    for (int avi = 0; avi < 8; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    if (Arg_print_time) {
        cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
    }
}

void doConv2D(
        int batch, int ih, int iw, int ic, 
        int kh, int kw, int od,
        int oh, int ow,
        Tensor<float>& padded_tensor, Tensor<float>& ker_tensor, Tensor<float>& out_tensor)
{
    clock_t start_c = clock();
    float* padded_val_ptr = (float*) padded_tensor.valPtr();
    float* ker_val_ptr = (float*) ker_tensor.valPtr();
    float* out_val_ptr = (float*) out_tensor.valPtr();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    float acc = 0;
                    __m256 r_av = _mm256_setzero_ps();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 8; c += 8) {
                                __m256 in_av = _mm256_loadu_ps(padded_val_ptr + i_idx + c);
                                __m256 k_av = _mm256_loadu_ps(ker_val_ptr + k_idx + c);
                                __m256 mu_av = _mm256_mul_ps(in_av, k_av);
                                r_av = _mm256_add_ps(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    float* r_av_ptr = (float*)&r_av;
                    for (int avi = 0; avi < 8; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    if (Arg_print_time) {
        cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
    }
}

// [DoConv2D] End.


void* threadFuncInt16(void* thread_arg) 
{
    ThreadArg<int16_t>* arg = (ThreadArg<int16_t>*) thread_arg;

    int batch = arg->padded_tensor->dim[0];
    int ih = arg->padded_tensor->dim[1];
    int iw = arg->padded_tensor->dim[2];
    int ic = arg->padded_tensor->dim[3];
    int kh = arg->ker_tensor->dim[0];
    int kw = arg->ker_tensor->dim[1];
    int od = arg->ker_tensor->dim[2];
    int oh = arg->out_tensor->dim[1];
    int ow = arg->out_tensor->dim[2];

    int16_t* padded_val_ptr = (int16_t*) arg->padded_tensor->valPtr();
    int16_t* ker_val_ptr = (int16_t*) arg->ker_tensor->valPtr();
    int16_t* out_val_ptr = (int16_t*) arg->out_tensor->valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = arg->oh_s; i < arg->oh_e; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int16_t acc = 0;
                    __m256i r_av = _mm256_setzero_si256();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 16; c += 16) {
                                __m256i in_av = _mm256_loadu_si256((__m256i*) (padded_val_ptr + i_idx + c));
                                __m256i k_av = _mm256_loadu_si256((__m256i*) (ker_val_ptr + k_idx + c));
                                __m256i mu_av = _mm256_mullo_epi16(in_av, k_av);
                                r_av = _mm256_adds_epi16(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    int16_t* r_av_ptr = (int16_t*)&r_av;
                    for (int avi = 0; avi < 16; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    return 0;
}

void* threadFuncInt32(void* thread_arg) 
{
    ThreadArg<int32_t>* arg = (ThreadArg<int32_t>*) thread_arg;

    int batch = arg->padded_tensor->dim[0];
    int ih = arg->padded_tensor->dim[1];
    int iw = arg->padded_tensor->dim[2];
    int ic = arg->padded_tensor->dim[3];
    int kh = arg->ker_tensor->dim[0];
    int kw = arg->ker_tensor->dim[1];
    int od = arg->ker_tensor->dim[2];
    int oh = arg->out_tensor->dim[1];
    int ow = arg->out_tensor->dim[2];

    int32_t* padded_val_ptr = (int32_t*) arg->padded_tensor->valPtr();
    int32_t* ker_val_ptr = (int32_t*) arg->ker_tensor->valPtr();
    int32_t* out_val_ptr = (int32_t*) arg->out_tensor->valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = arg->oh_s; i < arg->oh_e; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    int32_t acc = 0;
                    __m256i r_av = _mm256_setzero_si256();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 8; c += 8) {
                                __m256i in_av = _mm256_loadu_si256((__m256i*) (padded_val_ptr + i_idx + c));
                                __m256i k_av = _mm256_loadu_si256((__m256i*) (ker_val_ptr + k_idx + c));
                                __m256i mu_av = _mm256_mullo_epi32(in_av, k_av);
                                r_av = _mm256_add_epi32(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    int32_t* r_av_ptr = (int32_t*)&r_av;
                    for (int avi = 0; avi < 8; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    return 0;
}

void* threadFuncFloat(void* thread_arg)
{
    ThreadArg<float>* arg = (ThreadArg<float>*) thread_arg;

    int batch = arg->padded_tensor->dim[0];
    int ih = arg->padded_tensor->dim[1];
    int iw = arg->padded_tensor->dim[2];
    int ic = arg->padded_tensor->dim[3];
    int kh = arg->ker_tensor->dim[0];
    int kw = arg->ker_tensor->dim[1];
    int od = arg->ker_tensor->dim[2];
    int oh = arg->out_tensor->dim[1];
    int ow = arg->out_tensor->dim[2];

    float* padded_val_ptr = (float*) arg->padded_tensor->valPtr();
    float* ker_val_ptr = (float*) arg->ker_tensor->valPtr();
    float* out_val_ptr = (float*) arg->out_tensor->valPtr();

    for (int b = 0; b < batch; ++b) {
        for (int i = arg->oh_s; i < arg->oh_e; ++i) {
            for (int j = 0; j < ow; ++j) {
                for (int d = 0; d < od; ++d) {
                    float acc = 0;
                    __m256 r_av = _mm256_setzero_ps();
                    for (int di = 0; di < kh; ++di) {
                        for (int dj = 0; dj < kw; ++dj) {
                            int i_idx = b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic;
                            int k_idx = di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic;
                            int c = 0;
                            for (c = 0; c <= ic - 8; c += 8) {
                                __m256 in_av = _mm256_loadu_ps(padded_val_ptr + i_idx + c);
                                __m256 k_av = _mm256_loadu_ps(ker_val_ptr + k_idx + c);
                                __m256 mu_av = _mm256_mul_ps(in_av, k_av);
                                r_av = _mm256_add_ps(r_av, mu_av);
                            }
                            if (c < ic) {
                                for (; c < ic; ++c) {
                                    acc += padded_val_ptr[i_idx + c]
                                           * ker_val_ptr[k_idx + c];
                                }
                            }
                        }
                    }
                    float* r_av_ptr = (float*)&r_av;
                    for (int avi = 0; avi < 8; ++avi) {
                        acc += r_av_ptr[avi];
                    }
                    out_val_ptr[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }

    return 0;
}

template <typename T>
void doConv2Dpthread(int oh,
        Tensor<T>& padded_tensor, Tensor<T>& ker_tensor, Tensor<T>& out_tensor)
{
    clock_t start_c = clock();
    pthread_t threads[P_THREADS];
    ThreadArg<T> t_args[P_THREADS];

    int num_threads = min(P_THREADS, oh);
    int oh_part_size = oh / num_threads;

    t_args[0].padded_tensor = &padded_tensor;
    t_args[0].ker_tensor = &ker_tensor;
    t_args[0].out_tensor = &out_tensor;

    int t_id = -1;
    for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
        if (t_idx > 0) {
            t_args[t_idx] = t_args[0];
        }

        int oh_s = oh_part_size * t_idx;
        int oh_e = t_idx < num_threads - 1 ? oh_s + oh_part_size : oh;

        t_args[t_idx].oh_s = oh_s;
        t_args[t_idx].oh_e = oh_e;
        if (Arg_mode == 0) {
            t_id = pthread_create(&threads[t_idx], NULL, threadFuncFloat, (void*) &t_args[t_idx]);
        } else if (Arg_mode == 32) {
            t_id = pthread_create(&threads[t_idx], NULL, threadFuncInt32, (void*) &t_args[t_idx]);
        } else if (Arg_mode == 16) {
            t_id = pthread_create(&threads[t_idx], NULL, threadFuncInt16, (void*) &t_args[t_idx]);
        }
        if (t_id < 0) {
            perror("pthread error");
            exit(0);
        }
    }

    for (int t_idx = 0; t_idx < num_threads; ++t_idx) {
        pthread_join(threads[t_idx], NULL);
    }

    if (Arg_print_time) {
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
    // int ic = in_tensor.dim[3];

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
    
    doConv2Dpthread(oh,
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
    Tensor<T> padded_tensor = getQuantized<T>(s_in, unquan_padded_tensor);
    Tensor<T> quan_ker_tensor = getQuantized<T>(s_ker, ker_tensor);

    Tensor<T> out_tensor;
    out_tensor.dim[0] = batch;
    out_tensor.dim[1] = oh;
    out_tensor.dim[2] = ow;
    out_tensor.dim[3] = od;
    out_tensor.val.assign(batch * oh * ow * od, 0);

    doConv2D(
            batch, ih, iw, ic, kh, kw, od, oh, ow,
            padded_tensor, quan_ker_tensor, out_tensor);

    return getDequantized(s_in * s_ker, out_tensor);
}

bool initArgs(int argc, char* argv[]) {
    Arg_print_time = false;
    Arg_mode = 0;
    Arg_s_in = 0;
    Arg_s_ker = 0;

    int op_c;
    while ((op_c = getopt(argc, argv, "pi:k:")) != -1) {
        if (op_c == 'p') {
            Arg_print_time = true;
        } else if (op_c == 'i') {
            Arg_s_in = atof(optarg);
        } else if (op_c == 'k') {
            Arg_s_ker = atof(optarg);
        } else {
            return false;
        }
    }

    int op_i = optind;
    if (op_i + 2 >= argc) {
        return false;
    }
    Arg_in_fname = argv[op_i];
    Arg_ker_fname = argv[op_i + 1];
    string mode_str(argv[op_i + 2]);
    if (mode_str == "FP32") {
        Arg_mode = 0;
    } else if (mode_str == "INT32") {
        Arg_mode = 32;
    } else if (mode_str == "INT16") {
        Arg_mode = 16;
    } else {
        return false;
    }

    if (Arg_mode == 32 && (Arg_s_in == 0 || Arg_s_ker == 0)) {
        Arg_s_in = 15.59f;
        Arg_s_ker = 5368759.11f;
    } else if (Arg_mode == 16 && (Arg_s_in == 0 || Arg_s_ker == 0)) {
        Arg_s_in = 12.17f; 
        Arg_s_ker = 132.0f;
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

    constexpr char out_fname[] = "output_tensor.bin";
    if (Arg_mode == 0) {
        writeFile(out_fname, conv2D(in_tensor, ker_tensor));
    } else if (Arg_mode == 32) {
        writeFile(out_fname, quanConv2D<int32_t>(Arg_s_in, Arg_s_ker, in_tensor, ker_tensor));
    } else if (Arg_mode == 16) {
        writeFile(out_fname, quanConv2D<int16_t>(Arg_s_in, Arg_s_ker, in_tensor, ker_tensor));
    } else {
        assert(0);
    }
}