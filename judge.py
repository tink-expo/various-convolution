import numpy as np
import sys
import os
import math
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
    layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
            input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
    y = layer(inp)
    y_ans = K.eval(y)

    return y_ans

def shift_ans(in_bin, ker_bin):
    dim1, inp = read_file(in_bin)
    dim, ker = read_file(ker_bin)
    ker = ker.transpose(0, 1, 3, 2)
    dim[2], dim[3] = dim[3], dim[2]
    layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
            input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
    y = layer(inp)
    y_ans = K.eval(y)

    return y_ans


def NRMSE(x, y):
    sig = (np.square(x - y)).sum()
    numer = math.sqrt(sig / x.size)
    denom = y.max() - y.min()
    return numer / denom

def cmp_all():

    pwd = os.getcwd()
    out_bin = '{}/output_tensor.bin'.format(pwd)
    conv = '{}/probs/prob2/convolution'.format(pwd)
    for i in range(1, 4):
        in_bin = '{}/group2/{}/it.bin'.format(pwd, i)
        ker_bin = '{}/group2/{}/kt.bin'.format(pwd, i)
        os.system('{} {} {} {} {}'.format(
                conv, in_bin, ker_bin, sys.argv[1], sys.argv[2]))

        ans_bin = '{}/group2/ans/o{}.bin'.format(pwd, i)
        _, ans = read_file(ans_bin)
        # ans = shift_ans(in_bin, ker_bin)  # Judge with keras
        _, oup = read_file(out_bin)

        print('AVG: {}'.format(abs(ans).mean()))
        print('DIFF: {}'.format(abs(oup - ans).mean()))
        print('NRMSE: {}'.format(NRMSE(oup, ans)))
        print()
# pwd = os.getcwd()
# out_bin = '{}/output_tensor.bin'.format(pwd)
# _, oup = read_file(out_bin)
# print(abs(oup).sum() / oup.size)

def quan(arr, s, dt):
    q = np.array(arr * s, dtype=dt)
    print(arr[0, 0, :4, :4])
    print()
    print(q[0, 0, :4, :4])

def quan_i(s, dt):
    pwd = os.getcwd()
    i = 1
    in_bin = '{}/group2/{}/it.bin'.format(pwd, i)
    ker_bin = '{}/group2/{}/kt.bin'.format(pwd, i)
    _, inp = read_file(in_bin)
    quan(inp, s, dt)
    _, ker = read_file(ker_bin)
    quan(ker, s, dt)


cmp_all()

        




