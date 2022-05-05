import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from pedalboard import load_plugin

class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(
        self, frame_length, num_fft, hop_length, num_mels, sample_rate, f_min, f_max, eps, norm=False, **kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self.num_freqs = num_fft // 2 + 1
        self.norm = norm
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mels,
            num_spectrogram_bins=self.num_freqs,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        stfts = tf.signal.stft(
            input,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            fft_length=self.num_fft,
            pad_end=False,
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(
            tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0]
        )
        log_melgrams = _tf_log10(melgrams + self.eps)
        log_melgrams = tf.expand_dims(log_melgrams, 3)
    
            
        return log_melgrams

    def get_config(self):
        config = {
            'frame_length': self.frame_length,
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'eps': self.eps,
            'norm': self.norm
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
    
def rademacher(shape):
    x = np.random.binomial(1, .5, shape)
    x[x==0] = -1
    return x

def uniform(shape):
    return np.random.uniform(-1,1,shape)

def set_parameter(vst, parameter_idx, parameter_value):
    for i, key in enumerate(vst.parameters.keys()):
        if i == parameter_idx:
            setattr(vst, key, parameter_value*100.0)

def rem_nans(x):
    if np.isnan(x).any():
        x = np.nan_to_num(x, copy=False, nan=100)
    return np.clip(x, -100, 100)    
            
def forward(signal, params, param_map, vst, sample_rate):
    vst.reset()
    for i in range(len(params)):
        set_parameter(vst, param_map[i], params[i])
    ret = vst.process(signal, sample_rate)
    return rem_nans(ret)

def grad_batch_item(dye, xe, ye, vst, param_map, sample_rate, epsilon):
    vecJxe = np.ones_like(xe)
    vecJye = np.zeros_like(ye)
    c_k = epsilon
    delta_k = rademacher(ye.shape)
    J_plus = forward(xe, np.clip(ye + c_k * delta_k, 0.0, 1.0), param_map, vst, sample_rate)
    J_minus = forward(xe, np.clip(ye - c_k * delta_k, 0.0, 1.0), param_map, vst, sample_rate)
    grady_num = (J_plus - J_minus)
    for i in range(len(ye)):
        grady = grady_num / (2 * c_k * delta_k[i])
        vecJye[i] = np.dot(np.transpose(dye), grady)

    return vecJxe, vecJye

## VST Processor layer
class VSTProcessor(tfkl.Layer):
    def __init__(self, path_to_vst, sample_rate, n_processors, epsilon, 
            parameter_map, non_learnable_parameters, *args, **kwargs):
        super(VSTProcessor, self).__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.path_to_vst = path_to_vst
        self.n_processors = n_processors
        self.epsilon = epsilon

        self.mb = Parallel_Batch(self.sample_rate, self.path_to_vst, self.n_processors, self.epsilon,
            parameter_map, non_learnable_parameters)


    @tf.custom_gradient
    def process(self, signal, parameters):
        processed_audio = self.forward_batch(signal, parameters)  
        def gradient(dy):
            dy = np.array(dy)
            return self.run_gradient(dy, signal, parameters)

        return processed_audio, gradient
        
    def build(self, input_shape):
        super(VSTProcessor, self).build(input_shape)
        self.vst_func = setup_custom_vst_op(self.mb)
    
    def call(self, inputs, training=None):
        x = inputs[0]
        parameters = inputs[1]
        ret = tf.py_function(func=self.vst_func, inp=[x, parameters], Tout=tf.float32)
        ret.set_shape(x.get_shape())
        return ret
    
    def get_config(self):
        config = super(VSTProcessor, self).get_config()
        return config

    def print_current_parameters(self):
        self.mb.print_current_parameters()


class Parallel_Batch:
    def __init__(self, sample_rate, path_to_vst, n_processors, epsilon, 
            parameter_map, non_learnable_parameters={}):
        self.sample_rate = sample_rate
        self.n_processors = n_processors
        self.epsilon = epsilon
        self.path_to_vst = path_to_vst
        self.parameter_map = parameter_map
        self.non_learnable_parameters = non_learnable_parameters
        self.vsts = {}

    def init(self):
        self.vsts = {}
        for i in range(self.n_processors):
            self.vsts[i] = load_plugin(self.path_to_vst)
            if self.non_learnable_parameters:
                for k in self.non_learnable_parameters:
                    set_parameter(self.vsts[i], k, self.non_learnable_parameters[k])

    def run_grad_batch(self, dy, x, y):
        vecJx = np.empty_like(x)
        vecJy = np.empty_like(y)
        for i in range(self.n_processors):
            vecJx[i], vecJy[i] = grad_batch_item(dy[i], x[i], y[i], self.vsts[i], self.parameter_map, self.sample_rate, self.epsilon)
        return vecJx, vecJy

    def run_forward_batch(self, x, y):
        z = np.empty_like(x)
        for i in range(x.shape[0]):
            z[i] = forward(x[i], y[i], self.parameter_map, self.vsts[i], self.sample_rate)
        return z

    def print_current_parameters(self): 
        parameters = self.vsts[0].parameters
        for _, idx in enumerate(self.parameter_map.values()):
            for i, key in enumerate(parameters.keys()):
                if i == idx:
                    print('\n', key, ':', parameters[key])

def setup_custom_vst_op(mb):
    mb.init()

    @tf.custom_gradient
    def custom_grad_numeric_batch(x,y):
        x = np.array(x)
        y = np.array(y)

        def grad_batch(dy):
            dy = np.array(dy)
            return mb.run_grad_batch(dy, x, y)

        z = mb.run_forward_batch(x, y)

        return z, grad_batch

    return custom_grad_numeric_batch