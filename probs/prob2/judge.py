import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

def read_file(fname):
    whole = np.fromfile(fname)
    dim = np.frombuffer(whole.data, dtype=np.int32, count=4)
    val = np.frombuffer(whole.data, dtype=np.float32, offset=16)
    val = val.reshape(dim[0], dim[1], dim[2], dim[3])
    return dim, val

def read_fsize(fname):
    dim = np.fromfile(fname, dtype=np.int32, count=4)
    print(dim)

def get_ans(in_bin, layer_name):
    model = tf.keras.applications.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000
    )

    dim, inp = read_file(in_bin)

    for layer in model.layers:
        if layer.name == layer_name:
            y = layer(inp)
            y_ans = K.eval(y)

    return y_ans

def manual_ans(in_bin, ker_bin):
    dim1, inp = read_file(in_bin)
    dim, ker = read_file(ker_bin)
    print(dim1)
    print(dim)
    layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
            input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
    y = layer(inp)
    y_ans = K.eval(y)
    print(y_ans.shape)

    return y_ans

def shift_ans(in_bin, ker_bin):
    dim1, inp = read_file(in_bin)
    dim, ker = read_file(ker_bin)
    print(dim1)
    print(dim)
    ker = ker.transpose(0, 1, 3, 2)
    dim[2], dim[3] = dim[3], dim[2]
    layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
            input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
    y = layer(inp)
    y_ans = K.eval(y)

    return y_ans

in_bin = sys.argv[1]
out_bin = sys.argv[2]
ker_bin = sys.argv[3]
dim, oup = read_file(out_bin)
y_ans = shift_ans(in_bin, ker_bin)
print(y_ans.shape)
print(oup.shape)
print(y_ans.shape == oup.shape)
print(abs(y_ans - oup).max())
        




