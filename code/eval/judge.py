import numpy as np
import sys
import os
import math
import tensorflow as tf
import tensorflow.keras.backend as K
import copy
import random
import io

SEND = 10
DEVICE = '/gpu:1'

cwd = os.getcwd()
def get_out_bin():
    return '{}/output_tensor.bin'.format(cwd)
def get_conv_bin(prob_no):
    return '{}/../prob{}/convolution'.format(cwd, prob_no)
def get_in_bin(i):
    return '{}/../../group2/{}/input_tensor.bin'.format(cwd, i)
def get_ker_bin(i):
    return '{}/../../group2/{}/kernel_tensor.bin'.format(cwd, i)
# def get_in_bin(i):
#     return '{}/../../group1/{}/input_tensor.bin'.format(cwd, i)
# def get_ker_bin(i):
#     return '{}/../../group1/{}/kernel_tensor.bin'.format(cwd, i)
def get_ans_bin(ans_no, i):
    return '{}/../../group2o/{}/o{}.bin'.format(cwd, i, ans_no)

def read_file(fname):
    whole = np.fromfile(fname)
    dim = np.frombuffer(whole.data, dtype=np.int32, count=4)
    val = np.frombuffer(whole.data, dtype=np.float32, offset=16)
    val = val.reshape(dim[0], dim[1], dim[2], dim[3])
    return dim, val

def read_fsize(fname):
    dim = np.fromfile(fname, dtype=np.int32, count=4)
    print(dim)

def NRMSE(x, y):
    numer = math.sqrt((np.square(x - y)).mean())
    denom = y.max() - y.min()
    return numer / denom


## Judge

def keras_ans_r(in_bin, ker_bin):
    dim1, inp = read_file(in_bin)
    dim, ker = read_file(ker_bin)
    dim[2], dim[3] = dim[3], dim[2]
    ker = ker.reshape(dim)
    with tf.device(DEVICE):
        layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
                input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
        y = layer(inp)
        y_ans = K.eval(y)

    return y_ans

def keras_ans(in_bin, ker_bin):
    dim1, inp = read_file(in_bin)
    dim, ker = read_file(ker_bin)
    ker = ker.transpose(0, 1, 3, 2)
    dim[2], dim[3] = dim[3], dim[2]
    with tf.device(DEVICE):
        layer = tf.keras.layers.Conv2D(dim[3], (dim[0], dim[1]), padding='same', use_bias=False, 
                input_shape=(dim1[0], dim1[1], dim1[2], dim1[3]), weights=[ker])
        y = layer(inp)
        y_ans = K.eval(y)

    return y_ans

def cmp_fname_keras(in_bin, ker_bin, out_bin, r_flag):
    if r_flag:
        ans = keras_ans_r(in_bin, ker_bin)
    else:
        ans = keras_ans(in_bin, ker_bin)

    _, oup = read_file(out_bin)

    print('AVG: {}'.format(abs(ans).mean()))
    print('DIFF: {}'.format(abs(oup - ans).mean()))
    print('NRMSE: {}'.format(NRMSE(oup, ans)))


def cmp_prob(prob_no, ans_no, arg_lst, r_flag, use_keras, run_cpp_nrmse=False):
    assert(use_keras or not r_flag)
    out_bin = get_out_bin()
    conv = get_conv_bin(prob_no)
    for i in range(1, 4):
        in_bin = get_in_bin(i)
        ker_bin = get_ker_bin(i)
        read_fsize(in_bin)
        read_fsize(ker_bin)

        if len(arg_lst) == 0:
            cmd = '{} {} {}'.format(conv, in_bin, ker_bin)
        elif len(arg_lst) >= 3:
            cmd = '{} {} {} {} -i {} -k {}'.format(
                    conv, in_bin, ker_bin, *arg_lst[:3])
        else:
            cmd = '{} {} {} {}'.format(
                    conv, in_bin, ker_bin, arg_lst[0])

        if r_flag:
            cmd += ' -r'
        # cmd += ' -p c'
        print(cmd)
        os.system(cmd)

        ans_bin = get_ans_bin(ans_no, i)
        
        if use_keras:
            if r_flag:
                ans = keras_ans_r(in_bin, ker_bin)
            else:
                ans = keras_ans(in_bin, ker_bin)
        else:
            _, ans = read_file(ans_bin)

        odim, oup = read_file(out_bin)
        print(odim)

        print('AVG: {}'.format(abs(ans).mean()))
        print('DIFF: {}'.format(abs(oup - ans).mean()))
        print('NRMSE: {}'.format(NRMSE(oup, ans)))

        if run_cpp_nrmse:
            nrmse_cmd = './nrmse {} {}'.format(out_bin, ans_bin)
            os.system(nrmse_cmd)

        print()

def cmp_all(r_flag, use_keras, run_cpp_nrmse=False):
    def cmp_prob_argls(prob_no, ans_no, argls):
        for argl in argls:
            cmp_prob(prob_no, ans_no, argl, r_flag, use_keras, run_cpp_nrmse)
    cmp_prob_argls(1, 1, [
        []
    ])
    cmp_prob_argls(2, 1, [
        ['32'], ['16'], ['8']
    ])
    cmp_prob_argls(3, 3, [
        ['FP32'], ['INT32'], ['INT16']
    ])
    cmp_prob_argls(4, 4, [
        []
    ])


## Alternating Variable Method Search

def eval_error(prob_no, mode, answers, vector, vec_idx, vec_val):
    out_bin = get_out_bin()
    conv = get_conv_bin(prob_no)
    errors = []
    vec_org = vector[vec_idx]
    vector[vec_idx] = vec_val
    for i in range(1, 4):
        in_bin = get_in_bin(i)
        ker_bin = get_ker_bin(i)
        cmd = '{} {} {} {} -i {} -k {}'.format(
                conv, in_bin, ker_bin, mode, vector[0] / SEND, vector[1] / SEND)
        os.system(cmd)
        _, oup = read_file(out_bin)
        ans = answers[i]
        errors.append(NRMSE(oup, ans))
    vector[vec_idx] = vec_org
    return sum(errors) / len(errors)

def variable_search(prob_no, mode, answers, ini_vec, vec_idx):
    vec = [SEND * e for e in ini_vec]
    x = vec[vec_idx]
    fit = eval_error(prob_no, mode, answers, vec, vec_idx, x)
    if fit == 0:
        return [x / SEND for x in vec], 0

    while True:
        decr = eval_error(prob_no, mode, answers, vec, vec_idx, x - 1)
        incr = eval_error(prob_no, mode, answers, vec, vec_idx, x + 1)

        if fit <= decr and fit <= incr:
            break

        k = 1 if decr > incr else -1
        while True:
            fit_next = eval_error(prob_no, mode, answers, vec, vec_idx, x + k)
            if fit_next >= fit:
                break
            else:
                x = x + k
                k = 2 * k
                fit = fit_next

    vec[vec_idx] = x
    return [x / SEND for x in vec], fit

def avm_search(prob_no, ans_no, mode, tot_it, var_it):
    def get_initial_vec():
        if mode == '32' or mode == 'INT32':
            return [-random.randint(1, 20), random.randint(1e+6, 1e+7)]
        elif mode == '16' or mode == 'INT16':
            return [-random.randint(1, 20), random.randint(50, 200)]
        elif mode == '8':
            return [-(random.rand() + 0.1), random.randint(10, 100)]
        else:
            raise ValueError
    
    answers = {}
    for i in range(1, 4):
        ans_bin = get_ans_bin(ans_no, i)
        _, answers[i] = read_file(ans_bin)

    min_fit = 100
    for i in range(tot_it):
        fit = 100
        vec = get_initial_vec()
        for j in range(var_it):
            vec_idx = (j + 1) % 2
            new_vec, new_fit = variable_search(prob_no, mode, answers, vec, vec_idx)
            print(new_vec, new_fit)
            if new_fit < fit:
                vec = new_vec
                fit = new_fit
        if fit < min_fit:
            min_fit = fit
            min_vec = vec

    print(min_vec, min_fit)

if __name__=="__main__":
    pass