#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace std;

struct Tensor {
    string raw;
    int* dim;
    float* value;
};

bool readFile(const string& fname, Tensor* tensor)
{
    ifstream ifs(fname, ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    ifs.seekg(0, ios::end);
    tensor->raw.reserve(ifs.tellg());
    ifs.seekg(0, ios::beg);
    tensor->raw.assign((istreambuf_iterator<char>(ifs)),
            istreambuf_iterator<char>());

    tensor->dim = reinterpret_cast<int*>(const_cast<char*>(tensor->raw.data()));
    tensor->value = reinterpret_cast<float*>(const_cast<char*>(tensor->raw.data() + 16));
}


int main(int argc, char* argv[])
{
    Tensor t;
    if (readFile(string(argv[1]), &t)) {
        for (int i = 0; i < 4; ++i) {
            cout << t.dim[i] << endl;
        }
        cout << t.raw.size() << endl;
    }
}