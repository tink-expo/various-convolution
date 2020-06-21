#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <unistd.h>
#include <cassert>
#include <algorithm>
#include <cmath>

using namespace std;

// Global args.
char* Arg_x_fname;
char* Arg_y_fname;

template <typename T> 
struct Tensor {
    array<int, 4> dim;
    vector<T> val;

    inline T* valPtr()
    {
        return const_cast<T*>(val.data());
    }
};

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

double calcNrmse(const Tensor<float>& x_tensor,
        const Tensor<float>& y_tensor)
{
    assert(x_tensor.dim == y_tensor.dim);
    assert(x_tensor.val.size() > 0);
    double sq_diff_sum = 0.0;
    for (size_t i = 0; i < x_tensor.val.size(); ++i) {
        double diff = x_tensor.val[i] - y_tensor.val[i];
        sq_diff_sum += diff * diff;
    }
    double numer = sqrt(sq_diff_sum / x_tensor.val.size());
    double denom = 
            *max_element(y_tensor.val.begin(), y_tensor.val.end()) -
            *min_element(y_tensor.val.begin(), y_tensor.val.end());
    return numer / denom;
}

bool initArgs(int argc, char* argv[]) {
    int op_i = optind;
    if (op_i + 1 >= argc) {
        return false;
    }
    Arg_x_fname = argv[op_i];
    Arg_y_fname = argv[op_i + 1];
    return true;
}

int main(int argc, char* argv[])
{
    if (!initArgs(argc, argv)) {
        cout << "Invalid args." << endl;
        return 0;
    }

    Tensor<float> x_tensor;
    Tensor<float> y_tensor;

    if (!readFile(Arg_x_fname, x_tensor) || !readFile(Arg_y_fname, y_tensor)) {
        cout << "No such file for x_tensor or y_tensor." << endl;
        return 0;
    }

    if (x_tensor.dim != y_tensor.dim || x_tensor.val.empty()) {
        cout << "Invalid file(s)." << endl;
    }

    cout << calcNrmse(x_tensor, y_tensor) << endl;
}