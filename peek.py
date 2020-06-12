import numpy as np
import sys

def read_file(fname):
    whole = np.fromfile(fname)
    dim = np.frombuffer(whole.data, dtype=np.int32, count=4)
    val = np.frombuffer(whole.data, dtype=np.float32, offset=16)
    val = val.reshape(dim[0], dim[1], dim[2], dim[3])
    return dim, val

dim, val = read_file(sys.argv[1])
# print(val[0][0][:4][:4])
print(abs(val).sum() / val.size)