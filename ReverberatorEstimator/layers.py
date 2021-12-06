import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from pedalboard import load_plugin
import multiprocessing as mp
import sys

class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_fft, hop_length, num_mels, sample_rate, f_min, f_max, eps, **kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self.num_freqs = num_fft // 2 + 1
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
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        stfts = tf.signal.stft(
            input,
            frame_length=self.num_fft,
            frame_step=self.hop_length,
            pad_end=False,  # librosa test compatibility
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(  # assuming channel_first, so (b, c, f, t)
            tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0]
        )
        log_melgrams = _tf_log10(melgrams + self.eps)
        return tf.expand_dims(log_melgrams, 3)

    def get_config(self):
        config = {
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'eps': self.eps,
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))  

def setup_gradient_function(parallel_batch):
    parallel_batch.init()
    
    @tf.custom_gradient
    def process(signal, parameters):
        x = np.array(signal)
        params = np.array(parameters)     
        def gradient(dy):
            dy = np.array(dy)
            return parallel_batch.run_gradient_batch(dy, x, params)

        processed_audio = parallel_batch.run_forward_batch(x, params)
        return processed_audio, gradient
    return process

## VST Processor layer
class VSTProcessor(tfkl.Layer):
    def __init__(self, path_to_vst, sample_rate, n_processors=8, *args, **kwargs):
        super(VSTProcessor, self).__init__(*args, **kwargs)
        self.epsilon = 0.1
        self.parallel_batch = Parallel_Batch(n_processors, path_to_vst, sample_rate)
        
    def build(self, input_shape):
        super(VSTProcessor, self).build(input_shape)
        self.parallel_batch.init()
        self.gradient_function = setup_gradient_function(self.parallel_batch)
    
    def call(self, inputs, training=None):
        x = inputs[0]
        parameters = inputs[1]
        output_audio = tf.py_function(func=self.gradient_function, inp=[x, parameters], Tout=tf.float32)
        return output_audio
    
    def get_config(self):
        config = super(VSTProcessor, self).get_config()
        return config

def forward(vst, signal, params, sample_rate):
    print("Running forward function")
    sys.stdout.flush()
    print(vst)
    sys.stdout.flush()
#     vst.reset()
    print("VST reset")
    sys.stdout.flush()
    set_parameters(vst, params)
    print("Params set")
    sys.stdout.flush()
    output = vst.process(signal, sample_rate)
    print("Returning output of forward function")
    sys.stdout.flush()
    return output

def set_parameters(vst, parameters):
    params = np.copy(parameters)
    for i in range(len(params)):
        for j, key in enumerate(vst.parameters.keys()):
            if j == i:
                setattr(vst, key, params[i])

def run_gradient(dy, x, y, vst, c_k, sample_rate):
    vecJx = np.ones_like(x)
    vecJy = np.zeros_like(y)
    n_parameters = y.shape[0]
    delta_k = np.random.binomial(1, .5, n_parameters)
    delta_k[delta_k==0] = -1
    J_plus = forward(vst, x, np.clip(y + c_k*delta_k, 0.0, 1.0), sample_rate)
    J_minus = forward(vst, x, np.clip(y - c_k*delta_k, 0.0, 1.0), sample_rate)
    gradient_num = J_plus - J_minus
    for j in range(n_parameters):
        gradient = gradient_num / (2*c_k*delta_k[j])
        vecJy[j] = np.dot(np.transpose(dy), gradient)
    return vecJx, vecJy


class Parallel_Batch:
    def __init__(self, n_processors, vst_path, sample_rate):
        self.n_processors = n_processors
        self.vst_path = vst_path
        self.sample_rate = sample_rate
        self.vsts = {}
        self.procs = {}
        for i in range(self.n_processors):
            self.vsts[i] = load_plugin(self.vst_path)

    def init(self):
        print("Number of CPUs:", mp.cpu_count())
        procs = {}
        for i in range(self.n_processors):
            vst = self.vsts[i]
            q = mp.Queue()
            p = mp.Process(target=Parallel_Batch.queue_function, 
                args=(q, vst, self.sample_rate,))
            p.start()
            procs[i] = [p, q]

        self.procs = procs

    def __del__(self):
        for key in self.procs:
            self.procs[key][0].join() 
        self.plugins = {}    
        self.procs = {}

    def queue_function(q, vst, sample_rate):
        while True:
            msg, value = q.get()
            print(msg)
            print(value)
            sys.stdout.flush()
            try:
                if str(msg) == "grad":
                    dy, x, y = value
                    vecJx, vecJy = run_gradient(dy, x, y, vst, sample_rate)
                    q.put(vecJx, vecJy)
                elif str(msg) == "forward":
                    signal, params = value
                    print("Signal and params received")
                    sys.stdout.flush()
                    output = forward(vst, signal, params, sample_rate)
                    print("output")
                    sys.stdout.flush()
                    q.put(output)
                    print("output set")
                    sys.stdout.flush()
                elif str(msg) == "set_parameters":
                    parameters = value
                    set_parameters(vst, parameters)
                elif str(msg) == "reset":
                    vst.reset()
            except:
                pass
    
    def run_gradient_batch(self, dy, x, y):
        for i in range(dy.shape[0]):
            msg = ("grad", (dy[i], x[i], y[i]))
            self.procs[i][1].put(msg)
        
        vecJx = np.empty_like(x)
        vecJy = np.empty_like(y)
        for i in range(dy.shape[0]):
            vecJx[i], vecJy[i] = self.procs[i][1].get()
        return vecJx, vecJy

    def run_forward_batch(self, x, y):
        print("Running forward batch")
        for i in range(x.shape[0]):
            print("Forward %i" % i)
            msg = ("forward", (x[i], y[i]))
            self.procs[i][1].put(msg)
        output_batch = np.empty_like(x)
        for i in range(x.shape[0]):
            print("Receiving %i" % i)
            print(self.procs[i][1])
            output_batch[i] = self.procs[i][1].get()
            print("Done receiving %i" % i)
        return output_batch


    