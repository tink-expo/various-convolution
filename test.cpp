#include <ctime>
#include <vector>
#include <cstdint>
#include <iostream>
#include <immintrin.h>
#include <bitset>
#include <unistd.h>
using namespace std;

template <typename T>
void measure()
{
    int batch, ih, iw, ic, kh, kw, od, oh, ow, cs, as, bs;
    cin >> batch >> ih >> iw >> ic >> kh >> kw >> od >> oh >> ow >> cs >> as >> bs;

    T val = (T) 1.0f;
    cout << (int) val << endl;
    vector<T> av(as, val);
    vector<T> bv(bs, val);
    vector<T> cv(cs, val);

    clock_t start_c = clock();
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < od; ++d) {
            for (int i = 0; i < oh; ++i) {
                for (int j = 0; j < ow; ++j) {
                    T acc = 0;
                    for (int c = 0; c < ic; ++c) {
                        for (int di = 0; di < kh; ++di) {
                            for (int dj = 0; dj < kw; ++dj) {
                                acc += av[
                                    b * (ih * iw * ic)
                                    + (i + di) * (iw * ic)
                                    + (j + dj) * ic
                                    + c
                                ] * bv[
                                    di * (kw * od * ic)
                                    + dj * (od * ic)
                                    + d * ic
                                    + c
                                ];
                            }
                        }
                    }
                    cv[
                        b * (oh * ow * od)
                        + i * (ow * od)
                        + j * od
                        + d
                    ] = acc;
                }
            }
        }
    }
    for (int i = 0; i < 10; ++i) {
        cout << (int) cv[i] << " ";
    }
    cout << endl;
    cout << (double) (clock() - start_c) / CLOCKS_PER_SEC << endl;
}

void avx()
{
    __m256i a = _mm256_set_epi32(1 << 4, 1 << 11, 0, 0, 0, 0, 0, 0);
    __m256i b = _mm256_set_epi32(1 << 4, 1 << 11, 0, 0, 0, 0, 0, 0);
    __m256i c = _mm256_mullo_epi32(a, b);
    cout << hex;
    int* ap = (int*)&a;
    for (int i = 6; i < 8; ++i) {
        cout << bitset<32>(ap[i]) << " ";
    }
    cout << endl;
    int* cp = (int*)&c;
    for (int i = 6; i < 8; ++i) {
        cout << bitset<32>(cp[i]) << " ";
    }
    cout << endl;
}

main(int argc, char* argv[]) {
    int c;
    while ((c = getopt(argc, argv, "a:")) != -1) {
        if (c == 'a') {
            cout << optarg << endl;
        } else {
            abort();
        }
    }
    for (int i = optind; i < argc; ++i) {
        cout << argv[i] << endl;
    }
}