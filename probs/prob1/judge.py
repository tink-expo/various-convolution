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

model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000
)

dim, inp = read_file(sys.argv[1])
print(dim)

for layer in model.layers:
    if layer.name == 'conv3_block3_2_conv':
        y = layer(inp)
        print(y.shape)
        y_ans = K.eval(y)

dim, oup = read_file(sys.argv[2])

print(y_ans.shape == oup.shape)
print(abs(y_ans - oup).max())
        




