import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from pedalboard import load_plugin

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

## VST Processor layer
class VSTProcessor(tfkl.Layer):
    def __init__(self, path_to_vst, sample_rate, *args, **kwargs):
        super(VSTProcessor, self).__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.vst = load_plugin(path_to_vst)
        self.epsilon = 0.01
      
    def set_parameters(self, parameters):
        params = np.copy(parameters)
        for i in range(len(params)):
            for j, key in enumerate(self.vst.parameters.keys()):
                if j == i:
                    setattr(self.vst, key, params[i])
    
    def forward(self, signal, params):
        self.vst.reset()
        self.set_parameters(params)
        return self.vst.process(signal, self.sample_rate)

    def run_gradient(self, dy, x, y):
        use_fd = False
        vecJx = np.ones_like(x)
        vecJy = np.zeros_like(y)
        n_parameters = y.shape[0]
        if use_fd:
            for i in range(n_parameters):
                y[i] = np.clip(y[i] + self.epsilon, 0.0, 1.0)
                J_plus = self.forward(x, y)

                y[i] = np.clip(y[i] - (2*self.epsilon), 0.0, 1.0)
                J_minus = self.forward(x, y)

                gradient = (J_plus - J_minus)/(2.0*self.epsilon)
                y[i] = y[i] + 1*self.epsilon
                vecJy[i] = np.dot(np.transpose(dy), gradient)
            
        else: #use SPSA
            c_k = self.epsilon
            delta_k = np.random.binomial(1, .5, y.shape)
            delta_k[delta_k==0] = -1
            J_plus = self.forward(x, np.clip(y + c_k*delta_k, 0.0, 1.0))
            J_minus = self.forward(x, np.clip(y - c_k*delta_k, 0.0, 1.0))
            gradient_num = (J_plus - J_minus)
            for i in range(n_parameters):
                gradient = gradient_num / (2*c_k*delta_k[i])
                vecJy[i] = np.dot(np.transpose(dy), gradient)
                
                
        return vecJx, vecJy
            
    @tf.custom_gradient
    def process(self, signal: tf.Tensor, parameters: tf.Tensor) -> np.ndarray:
        x = np.array(signal)
        params = np.array(parameters) # batch size x num params  
        self.set_parameters(params[0])      
        def gradient(dy):
            dy = np.array(dy)
            # index 0 for batch size 1
            return self.run_gradient(dy[0], x[0], params[0])
            
        processed_audio = self.vst.process(x, self.sample_rate)
        processed_audio = tf.reshape(processed_audio, (1,96000))
            
        return processed_audio, gradient
        
    def build(self, input_shape):
        super(VSTProcessor, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs[0]
        parameters = inputs[1]
        output_audio = tf.py_function(func=self.process, inp=[x, parameters], Tout=tf.float32)
        return output_audio
    
    def get_config(self):
        config = super(VSTProcessor, self).get_config()
        return config