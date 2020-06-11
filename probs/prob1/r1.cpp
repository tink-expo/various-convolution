#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace std;

struct Tensor {
    int dim[4];
    std::vector<float> value;
};

std::string readFile(const std::string& fname)
{
    std::string ret;
    std::ifstream istr(fname, std::ios::binary);
    if (!istr.is_open()) {
        return ret;
    }

    istr >> ret;
    return ret;
}

// Tensor readTensor(const std::string& fdata)
// {
//     Tensor tensor;
//     memcpy(static_cast<void*>(tensor.dim),
//             static_cast<const void*>(fdata.data()),
//             sizeof(tensor.dim));

//     size_t dim_all = tensor.dim[0] * tensor.dim[1] * tensor.dim[2] * tensor.dim[3];
//     // tensor.value.resize(dim_all);
//     // memcpy(static_cast<void*>(tensor.value.data()),
//     //         static_cast<const void*>(fdata.data() + sizeof(tensor.dim)),
//     //         sizeof(float) * dim_all);
// }

Tensor readTensor(const std::string

int main(int argc, char* argv[])
{
    // Tensor tensor = readTensor(readFile(std::string(argv[1])));
    // std::cout << tensor.dim[0] << " " << tensor.dim[1] << " " << tensor.dim[2] << " " << tensor.dim[3] << std::endl;
    // std::cout << tensor.value.size() << std::endl;
    std::string str = readFile(std::string(argv[1]));
    for (int i = 0; i < 16; ++i) {
        std::cout << (unsigned int) str[i] << std::endl;
    }
    std::cout << str.size() << std::endl;
}