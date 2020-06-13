import numpy as np
import sys
import os
import math
import tensorflow as tf
import tensorflow.keras.backend as K
import copy
import random
import io
import subprocess

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
    numer = math.sqrt((np.square(x - y)).mean())
    denom = y.max() - y.min()
    return numer / denom

def cmp_all(prob_no, ans_no):
    pwd = os.getcwd()
    out_bin = '{}/output_tensor.bin'.format(pwd)
    conv = '{}/probs/prob{}/convolution'.format(pwd, prob_no)
    for i in range(1, 4):
        in_bin = '{}/group2/{}/it.bin'.format(pwd, i)
        ker_bin = '{}/group2/{}/kt.bin'.format(pwd, i)
        
        if sys.argv[1] == '0':
            cmd = '{} {} {} -p'.format(conv, in_bin, ker_bin)
        elif len(sys.argv) > 3:
            cmd = '{} {} {} {} -i {} -k {} -p'.format(
                    conv, in_bin, ker_bin, sys.argv[1], sys.argv[2], sys.argv[3])
        else:
            cmd = '{} {} {} {} -p'.format(
                    conv, in_bin, ker_bin, sys.argv[1])
        print(cmd)
        os.system(cmd)

        ans_bin = '{}/group2/{}/o{}.bin'.format(pwd, i, ans_no)
        _, ans = read_file(ans_bin)  # Judge with mine
        # ans = shift_ans(in_bin, ker_bin)  # Judge with keras
        _, oup = read_file(out_bin)

        print('AVG: {}'.format(abs(ans).mean()))
        print('DIFF: {}'.format(abs(oup - ans).mean()))
        print('NRMSE: {}'.format(NRMSE(oup, ans)))
        print()

SEND = 10

def eval_error(prob_no, mode, answers, vector, vec_idx, vec_val):
    pwd = os.getcwd()
    out_bin = '{}/output_tensor.bin'.format(pwd)
    conv = '{}/probs/prob{}/convolution'.format(pwd, prob_no)
    errors = []
    vec_org = vector[vec_idx]
    vector[vec_idx] = vec_val
    for i in range(1, 4):
        in_bin = '{}/group2/{}/it.bin'.format(pwd, i)
        ker_bin = '{}/group2/{}/kt.bin'.format(pwd, i)
        cmd = '{} {} {} {} -i {} -k {} -p'.format(
                conv, in_bin, ker_bin, mode, vector[0] / SEND, vector[1] / SEND)
        os.system(cmd)
        _, oup = read_file(out_bin)
        ans = answers[i]
        errors.append(NRMSE(oup, ans))
    vector[vec_idx] = vec_org
    return sum(errors) / len(errors)

def variable_search(mode, answers, ini_vec, vec_idx):
    vec = [SEND * e for e in ini_vec]
    x = vec[vec_idx]
    fit = eval_error(mode, answers, vec, vec_idx, x)
    if fit == 0:
        return [x / SEND for x in vec], 0

    while True:
        decr = eval_error(mode, answers, vec, vec_idx, x - 1)
        incr = eval_error(mode, answers, vec, vec_idx, x + 1)

        if fit <= decr and fit <= incr:
            break

        k = 1 if decr > incr else -1
        while True:
            fit_next = eval_error(mode, answers, vec, vec_idx, x + k)
            if fit_next >= fit:
                break
            else:
                x = x + k
                k = 2 * k
                fit = fit_next

    vec[vec_idx] = x
    return [x / SEND for x in vec], fit

def avm_search(prob_no, ans_no, mode):
    pwd = os.getcwd()
    answers = {}
    for i in range(1, 4):
        ans_bin = '{}/group2/{}/o{}.bin'.format(pwd, i, ans_no)
        _, answers[i] = read_file(ans_bin)

    min_fit = 100
    for i in range(1):
        fit = 100
        # vec = [random.choice([-1, 1]) * random.random(), random.randint(20, 200)]
        vec = [10.09, 5368759.11]
        for j in range(20):
            vec_idx = j % 2
            new_vec, new_fit = variable_search(mode, answers, vec, vec_idx)
            print(new_vec, new_fit)
            if new_fit < fit:
                vec = new_vec
                fit = new_fit
        if fit < min_fit:
            min_fit = fit
            min_vec = vec

    print(min_vec, min_fit)

def meas_time(prob_no, mode, runs):
    res = ''
    pwd = os.getcwd()
    conv = '{}/probs/prob{}/convolution'.format(pwd, prob_no)

    for i in range(1, 4):
        in_bin = '{}/group2/{}/it.bin'.format(pwd, i)
        ker_bin = '{}/group2/{}/kt.bin'.format(pwd, i)

        if mode == '0':
            cmd = '{} {} {} -p'.format(conv, in_bin, ker_bin)
        else:
            cmd = '{} {} {} {} -p'.format(
                    conv, in_bin, ker_bin, mode)

    #     for j in range(runs):
    #         mtime = subprocess.check_output(cmd, shell=True)
    #         mtime = mtime.decode('utf-8').strip()
    #         res += mtime + '\n'
    #     res += '\n'

    # print(res)
        print(i)
        for j in range(runs):
            os.system(cmd)
        print()


if __name__=="__main__":
    # avm_search(sys.argv[1])
    cmp_all(4, 4)
    # meas_time(4, '0', 10)
        




